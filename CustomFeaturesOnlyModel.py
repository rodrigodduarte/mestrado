import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

import torchvision
from torchvision import models
from torchvision.models.convnext import (convnext_tiny, ConvNeXt_Tiny_Weights,
                                         convnext_small, ConvNeXt_Small_Weights,
                                         convnext_base, ConvNeXt_Base_Weights,
                                         convnext_large, ConvNeXt_Large_Weights)
from torchvision.models.swin_transformer import (swin_t, Swin_T_Weights,
                                                 swin_s, Swin_S_Weights,
                                                 swin_b, Swin_B_Weights)
import torchmetrics
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
from torchmetrics.classification import MulticlassConfusionMatrix

import pytorch_lightning as pl

import config


class CustomFeaturesOnlyModel(pl.LightningModule):
    """
    Modelo para treinamento ISOLADO do vetor de características.
    Compatível com loaders que retornam:
      - (features, labels)
      - (images, features, labels)  -> imagens são ignoradas
      - {"features": tensor, "label": tensor}
    """

    def __init__(self,
                 name_dataset: str,
                 shape: tuple,
                 epochs: int,
                 learning_rate: float,
                 features_dim: int,
                 drop_path_rate: float,
                 num_classes: int,
                 label_smoothing: float,
                 optimizer_momentum: tuple,
                 weight_decay: float,
                 layer_scale: float,
                 auto_project: bool = True,
                 class_weights=None):
        super().__init__()

        self.save_hyperparameters(ignore=[
            "metric.goal", "metric.name", "parameters.batch_size",
            "parameters.layer_scale", "parameters.learning_rate.distribution",
            "parameters.learning_rate.max", "parameters.learning_rate.min"
        ])

        self.num_classes = num_classes
        self.features_dim = features_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_momentum = optimizer_momentum
        self.epochs = epochs
        self.auto_project = auto_project

        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy   = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy  = Accuracy(task='multiclass', num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1   = F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1  = F1Score(task="multiclass", num_classes=num_classes)

        self.train_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_precision   = Precision(task="multiclass", num_classes=num_classes)
        self.test_precision  = Precision(task="multiclass", num_classes=num_classes)

        self.train_recall = Recall(task="multiclass", num_classes=num_classes)
        self.val_recall   = Recall(task="multiclass", num_classes=num_classes)
        self.test_recall  = Recall(task="multiclass", num_classes=num_classes)

        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

        # Cabeça MLP
        adjusted_dim = self.features_dim
        scaled_dim   = int(max(int(adjusted_dim * layer_scale), max(64, num_classes)))

        self.input_norm = nn.LayerNorm(adjusted_dim, eps=1e-6, elementwise_affine=True)
        self.fc1 = nn.Linear(adjusted_dim, scaled_dim)
        self.mid_norm = nn.LayerNorm(scaled_dim, eps=1e-6, elementwise_affine=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc_out = nn.Linear(scaled_dim, num_classes)

        # Projeção automática opcional caso a dimensão não bata
        self.auto_proj = None  # criada sob demanda no primeiro forward, se necessário

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        cw = None
        if class_weights is not None:
            try:
                cw = torch.as_tensor(class_weights, dtype=torch.float32)
            except Exception:
                cw = None
        self.register_buffer("class_weights", cw if cw is not None else None)
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=float(label_smoothing)
        )

    # ----------------------- Utils -----------------------
    @staticmethod
    def _extract(batch):
        """Suporta (feats, y), (img, feats, y) ou dict({'features', 'label'})."""
        if isinstance(batch, dict):
            feats = batch.get('features', None)
            y = batch.get('label', None)
            if feats is None or y is None:
                raise ValueError("Batch dict deve conter chaves 'features' e 'label'.")
            return feats, y

        if not isinstance(batch, (tuple, list)):
            raise ValueError("Batch deve ser tupla/lista (features,label) ou dict {'features','label'}.")

        if len(batch) == 2:
            feats, y = batch
        elif len(batch) == 3:
            _img, feats, y = batch  # ignora imagens
        else:
            raise ValueError("Batch deve ter 2 ou 3 elementos: (features, labels) ou (images, features, labels).")
        return feats, y

    @staticmethod
    def _prepare_feats(feats):
        """
        Garante tensor float32 [B, D].
        Aceita [B, D], [B, D, 1], ou [D] (vira [1, D]).
        """
        if isinstance(feats, (list, tuple)):
            feats = torch.tensor(feats)
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats)

        feats = feats.float()
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)  # [D] -> [1, D]
        elif feats.ndim > 2:
            # Flatten do tail
            feats = feats.view(feats.size(0), -1)
        return feats

    @staticmethod
    def _prepare_labels(y):
        if isinstance(y, (list, tuple)):
            y = torch.tensor(y)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        return y.long().view(-1)

    # ----------------------- Forward -----------------------
    def forward(self, features):
        x = features
        if self.auto_proj and x.size(-1) != self.features_dim:
            # cria projeção 1x se necessário
            if self.auto_proj is None:
                self.auto_proj = nn.Linear(x.size(-1), self.features_dim).to(x.device)
            x = self.auto_proj(x)

        x = self.input_norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.mid_norm(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

    def _common_step(self, batch):
        feats, y = self._extract(batch)
        feats = self._prepare_feats(feats)
        y = self._prepare_labels(y)

        logits = self(feats)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        return loss, preds, y

    # ----------------------- Steps -----------------------
    def training_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch)
        self.train_accuracy(preds, y)
        self.train_f1(preds, y)
        self.train_precision(preds, y)
        self.train_recall(preds, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': self.train_accuracy},
                      prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch)
        self.val_accuracy(preds, y)
        self.val_f1(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.log_dict({'val_loss': loss, 'val_accuracy': self.val_accuracy},
                      prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch)
        self.test_accuracy(preds, y)
        self.test_f1(preds, y)
        self.test_precision(preds, y)
        self.test_recall(preds, y)
        self.test_confusion_matrix(preds, y)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy.compute(), prog_bar=True)
        self.log("test_f1",        self.test_f1.compute(),        prog_bar=True)
        self.log("test_precision", self.test_precision.compute(), prog_bar=True)
        self.log("test_recall",    self.test_recall.compute(),    prog_bar=True)

        return {
            "test_loss": loss,
            "test_accuracy": self.test_accuracy.compute(),
            "test_f1": self.test_f1.compute(),
            "test_precision": self.test_precision.compute(),
            "test_recall": self.test_recall.compute(),
        }

    def on_test_epoch_end(self):
        confm = self.test_confusion_matrix.compute().cpu().numpy()
        self.test_confusion_matrix.reset()
        self.train_accuracy.reset(); self.val_accuracy.reset(); self.test_accuracy.reset()
        self.train_f1.reset(); self.val_f1.reset(); self.test_f1.reset()
        self.train_precision.reset(); self.val_precision.reset(); self.test_precision.reset()
        self.train_recall.reset(); self.val_recall.reset(); self.test_recall.reset()
        print("✅ Matriz de Confusão calculada após o teste (features-only).")
        return confm

    # ----------------------- Optim -----------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay,
                                betas=self.optimizer_momentum)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        return {"optimizer": opt, "lr_scheduler": sch}
