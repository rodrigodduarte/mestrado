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
from compact_transform.src import cct_14_7x2_224, cct_14_7x2_384, cct_14_7x2_384_fl

class CustomModel(pl.LightningModule):
    def __init__(self, tmodel, name_dataset, shape, epochs, learning_rate,
                 drop_path_rate, num_classes, label_smoothing, optimizer_momentum,
                 weight_decay, layer_scale):
        
        super(CustomModel, self).__init__()

        self.save_hyperparameters(ignore=["method", "metric.goal", "metric.name","parameters.batch_size",
                                          "parameters.layer_scale", "parameters.learning_rate.distribution",
                                          "parameters.learning_rate.max", "parameters.learning_rate.min"])
        
        self.tmodel = tmodel
        self.name_dataset = name_dataset
        self.shape = shape
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.optimizer_momentum = optimizer_momentum
        self.weight_decay= weight_decay
        self.layer_scale = layer_scale
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        self.model_dim = 0
        self.validation_step_outputs = []
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)       
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes) 
        
        self.train_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=num_classes)
        
        self.train_recall = Recall(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)
        self.test_recall = Recall(task="multiclass", num_classes=num_classes)

        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)



        # self.dl_model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT, 
        #                                 drop_path_rate=self.drop_path_rate)
        
                # Escolha do modelo
        if tmodel == "convnext_t":
            self.model_dim = 768
            self.dl_model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT, 
                                            drop_path_rate=self.drop_path_rate)
            self.sequential_layers = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(self.model_dim, eps=1e-6, elementwise_affine=True),
            )
            self.dl_model.classifier = self.sequential_layers

        if tmodel == "swint_t":
            self.model_dim = 768
            self.dl_model = swin_t(weights=Swin_T_Weights.DEFAULT)
            self.sequential_layers = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(self.model_dim, eps=1e-6, elementwise_affine=True),
                )
            self.dl_model.head = self.sequential_layers

        # Modelo de combinação ajustado
        adjusted_dim = self.model_dim
        scaled_dim = int(adjusted_dim * self.layer_scale)

        self.model = nn.Sequential(
            nn.Linear(adjusted_dim, scaled_dim),
            nn.GELU(approximate='none'),
            nn.LayerNorm(scaled_dim),
            nn.Dropout(p=0.3),
            nn.Linear(scaled_dim, self.num_classes)
        )
        
        

    def forward(self, x):
        x = self.dl_model(x)
        x = self.model(x)
        return x


    def training_step(self, batch, batch_idx):
        images, labels, logits, loss, preds = self._commom_step(batch, batch_idx)

        # Calcular a precisão
        self.train_accuracy(preds, labels)
        
        # Logar a perda e a acurácia
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        # Retornar a perda para o processamento posterior
        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        images, labels, logits, loss, preds = self._commom_step(batch, batch_idx)

        # Calcular a precisão para validação
        self.val_accuracy(preds, labels)
        
        # Logar a perda e a acurácia no conjunto de validação
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True, on_epoch=True)
        
        # Retornar a perda e a acurácia
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        images, labels, logits, loss, preds = self._commom_step(batch, batch_idx)

        # Atualiza as métricas corretamente
        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)

        # Loga as métricas corretamente
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy.compute(), prog_bar=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True)
        self.log("test_precision", self.test_precision.compute(), prog_bar=True)
        self.log("test_recall", self.test_recall.compute(), prog_bar=True)

        return {
            "test_loss": loss,
            "test_accuracy": self.test_accuracy.compute(),
            "test_f1": self.test_f1.compute(),
            "test_precision": self.test_precision.compute(),
            "test_recall": self.test_recall.compute()        
            }
    
    def on_test_epoch_end(self):
        self.test_accuracy.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()

        # 🔹 Obter a matriz de confusão já acumulada pela métrica integrada
        conf_matrix_value = self.test_confusion_matrix.compute().cpu().numpy()
        self.test_confusion_matrix.reset()  # 🔹 Reseta a métrica para futuras execuções

        print("Matriz de Confusão calculada após o teste.")

        return conf_matrix_value


    def on_validation_epoch_end(self):
        # Aggregate predictions and perform analysis
        avg_loss = torch.mean(torch.tensor(self.validation_step_outputs))
        self.log('avg_val_loss', avg_loss)
        self.validation_step_outputs.clear()  # Clear outputs for the next epoch
        
    def _commom_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        preds = torch.argmax(logits, 1)

        return images, labels, logits, loss, preds

    def configure_optimizers(self):
        # Definir o otimizador com os grupos de parâmetros
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            betas = self.optimizer_momentum,
            weight_decay=self.weight_decay)

        # Definir o scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Retornar o otimizador e o scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Step the scheduler per epoch
                'monitor': 'val_loss',  # Optional, monitor val_loss (useful for other schedulers)
                'frequency': 1,  # Apply the scheduler every epoch
            }
        }


class CustomEnsembleModel(pl.LightningModule):
    def __init__(self, tmodel, name_dataset, shape, epochs, learning_rate, features_dim,
                 drop_path_rate, num_classes, label_smoothing, optimizer_momentum,
                 weight_decay, layer_scale):
        
        super(CustomEnsembleModel, self).__init__()

        self.save_hyperparameters(ignore=["method", "metric.goal", "metric.name","parameters.batch_size",
                                          "parameters.layer_scale", "parameters.learning_rate.distribution",
                                          "parameters.learning_rate.max", "parameters.learning_rate.min"])
        
        self.tmodel = tmodel
        self.name_dataset = name_dataset
        self.shape = shape
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.features_dim = features_dim
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.optimizer_momentum = optimizer_momentum
        self.weight_decay= weight_decay
        self.layer_scale = layer_scale
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        self.model_dim = 0
        self.validation_step_outputs = []
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)       
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes) 
        
        self.train_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=num_classes)
        
        self.train_recall = Recall(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)
        self.test_recall = Recall(task="multiclass", num_classes=num_classes)

        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)



        # self.dl_model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT, 
        #                                 drop_path_rate=self.drop_path_rate)
        
                # Escolha do modelo
        if tmodel == "convnext_t":
            self.model_dim = 768
            self.dl_model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT, 
                                            drop_path_rate=self.drop_path_rate)
            self.sequential_layers = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(self.model_dim, eps=1e-6, elementwise_affine=True),
            )
            self.dl_model.classifier = self.sequential_layers

        if tmodel == "swint_t":
            self.model_dim = 768
            self.dl_model = swin_t(weights=Swin_T_Weights.DEFAULT)
            self.sequential_layers = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(self.model_dim, eps=1e-6, elementwise_affine=True),
                )
            self.dl_model.head = self.sequential_layers

        # Modelo de combinação ajustado
        adjusted_dim = self.features_dim + self.model_dim
        scaled_dim = int(adjusted_dim * self.layer_scale)

        self.ensemble_model = nn.Sequential(
            nn.Linear(adjusted_dim, scaled_dim),
            nn.GELU(approximate='none'),
            nn.LayerNorm(scaled_dim),
            nn.Dropout(p=0.3),
            nn.Linear(scaled_dim, self.num_classes)
        )
        
        

    def forward(self, x, features):
        x = self.dl_model(x)
        x = torch.cat((x, features), dim=1)
        x = self.ensemble_model(x)
        return x


    def training_step(self, batch, batch_idx):
        images, features, labels, logits, loss, preds = self._commom_step(batch, batch_idx)

        # Calcular a precisão
        self.train_accuracy(preds, labels)
        
        # Logar a perda e a acurácia
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        # Retornar a perda para o processamento posterior
        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        images, features, labels, logits, loss, preds = self._commom_step(batch, batch_idx)

        # Calcular a precisão para validação
        self.val_accuracy(preds, labels)
        
        # Logar a perda e a acurácia no conjunto de validação
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True, on_epoch=True)
        
        # Retornar a perda e a acurácia
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        images, features, labels, logits, loss, preds = self._commom_step(batch, batch_idx)

        # Atualiza as métricas corretamente
        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)

        # Loga as métricas corretamente
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy.compute(), prog_bar=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True)
        self.log("test_precision", self.test_precision.compute(), prog_bar=True)
        self.log("test_recall", self.test_recall.compute(), prog_bar=True)

        return {
            "test_loss": loss,
            "test_accuracy": self.test_accuracy.compute(),
            "test_f1": self.test_f1.compute(),
            "test_precision": self.test_precision.compute(),
            "test_recall": self.test_recall.compute()        
            }
    
    def on_test_epoch_end(self):
        self.test_accuracy.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()

        # 🔹 Obter a matriz de confusão já acumulada pela métrica integrada
        conf_matrix_value = self.test_confusion_matrix.compute().cpu().numpy()
        self.test_confusion_matrix.reset()  # 🔹 Reseta a métrica para futuras execuções

        print("Matriz de Confusão calculada após o teste.")

        return conf_matrix_value


    def on_validation_epoch_end(self):
        # Aggregate predictions and perform analysis
        avg_loss = torch.mean(torch.tensor(self.validation_step_outputs))
        self.log('avg_val_loss', avg_loss)
        self.validation_step_outputs.clear()  # Clear outputs for the next epoch
        
    def _commom_step(self, batch, batch_idx):
        images, features, labels = batch
        logits = self.forward(images, features)
        loss = self.fn_loss(logits, labels)
        preds = torch.argmax(logits, 1)

        return images, features, labels, logits, loss, preds

    def configure_optimizers(self):
        # Definir o otimizador com os grupos de parâmetros
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            betas = self.optimizer_momentum,
            weight_decay=self.weight_decay)

        # Definir o scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Retornar o otimizador e o scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Step the scheduler per epoch
                'monitor': 'val_loss',  # Optional, monitor val_loss (useful for other schedulers)
                'frequency': 1,  # Apply the scheduler every epoch
            }
        }


class CustomModelTriple(pl.LightningModule):
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
                 layer_scale: float):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.features_dim = features_dim
        self.layer_scale = layer_scale
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_momentum = optimizer_momentum

        # === MÉTRICAS ===
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)       
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes) 
        
        self.train_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=num_classes)
        
        self.train_recall = Recall(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)
        self.test_recall = Recall(task="multiclass", num_classes=num_classes)

        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

        # === BACKBONES ===
        self.convnext_model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT,
                                                   drop_path_rate=drop_path_rate)
        self.convnext_model.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True)
        )

        self.swint_model = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.swint_model.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True)
        )

        # Saída combinada dos dois modelos de imagem
        self.image_dim = 1536

        # Camada final: (convnext + swin + vetor de características)
        adjusted_dim = self.image_dim + self.features_dim
        scaled_dim = int(adjusted_dim * layer_scale)
        
        self.ensemble_model = nn.Sequential(
            nn.Linear(adjusted_dim, scaled_dim),
            nn.GELU(),
            nn.LayerNorm(scaled_dim),
            nn.Dropout(0.3),
            nn.Linear(scaled_dim, num_classes)
        )
        
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, x, features):
        x_conv = self.convnext_model(x)
        x_swin = self.swint_model(x)
        x_img = torch.cat((x_conv, x_swin), dim=1)
        x_total = torch.cat((x_img, features), dim=1)
        return self.ensemble_model(x_total)

    def _common_step(self, batch, batch_idx):
        images, features, labels = batch
        logits = self.forward(images, features)
        loss = self.fn_loss(logits, labels)
        preds = torch.argmax(logits, 1)
        return images, features, labels, logits, loss, preds

    def training_step(self, batch, batch_idx):
        images, features, labels, logits, loss, preds = self._common_step(batch, batch_idx)
        self.train_accuracy(preds, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images, features, labels, logits, loss, preds = self._common_step(batch, batch_idx)
        self.val_accuracy(preds, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True, on_epoch=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        images, features, labels, logits, loss, preds = self._common_step(batch, batch_idx)
        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_confusion_matrix(preds, labels)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy.compute(), prog_bar=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True)
        self.log("test_precision", self.test_precision.compute(), prog_bar=True)
        self.log("test_recall", self.test_recall.compute(), prog_bar=True)

        return {
            "test_loss": loss,
            "test_accuracy": self.test_accuracy.compute(),
            "test_f1": self.test_f1.compute(),
            "test_precision": self.test_precision.compute(),
            "test_recall": self.test_recall.compute()
        }

    def on_test_epoch_end(self):
        self.test_accuracy.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        conf_matrix_value = self.test_confusion_matrix.compute().cpu().numpy()
        self.test_confusion_matrix.reset()
        print("✅ Matriz de Confusão calculada após o teste.")
        return conf_matrix_value

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay,
                                      betas=self.optimizer_momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class ReLuMLP2L(pl.LightningModule):
    def __init__(self, features_dim, num_classes,
                 hidden_factor=4, p_drop=0.3,
                 lr=5e-5, wd=1e-4, label_smoothing=0.0):
        super().__init__()
        h = hidden_factor * features_dim
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.LayerNorm(features_dim),
            nn.Linear(features_dim, h),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(h, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, feats):               # feats shape: [B, features_dim]
        return self.net(feats)

    def _step(self, batch):
        feats, y = batch                    # dataloader já entrega vetor, label
        logits = self(feats)
        loss = self.criterion(logits, y)
        preds = logits.argmax(1)
        return loss, preds, y

    def training_step(self, batch, _):
        loss, preds, y = self._step(batch)
        self.acc(preds, y)
        self.log_dict({"train_loss": loss,
                       "train_acc": self.acc},
                      prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, preds, y = self._step(batch)
        self.acc(preds, y)
        self.log_dict({"val_loss": loss,
                       "val_acc": self.acc},
                      prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sch}


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
                 auto_project: bool = True):
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
    

class CustomEnsembleModel_MLP2(pl.LightningModule):
    def __init__(self,
                 tmodel: str,                     # convnext | swin
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
                 layer_scale: float):
        super().__init__()
        self.save_hyperparameters()

        self.tmodel = tmodel.lower()
        self.num_classes = num_classes
        self.features_dim = features_dim
        self.layer_scale = layer_scale
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_momentum = optimizer_momentum

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


        if self.tmodel in {"convnext_t", "convnext", "convnext_tiny"}:
            self.backbone = models.convnext_tiny(
                weights=ConvNeXt_Tiny_Weights.DEFAULT,
                drop_path_rate=drop_path_rate
            )
            self.backbone.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(768, eps=1e-6, elementwise_affine=True)
            )
            backbone_dim = 768
        elif self.tmodel in {"swin_t", "swin", "swin_transformer"}:
            self.backbone = swin_t(weights=Swin_T_Weights.DEFAULT)
            self.backbone.head = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(768, eps=1e-6, elementwise_affine=True)
            )
            backbone_dim = 768
        else:
            raise ValueError(f"tmodel '{tmodel}' não suportado. Use 'convnext' ou 'swin'.")


        adjusted_dim = backbone_dim + self.features_dim
        hidden = int(adjusted_dim * layer_scale)

        self.ensemble_model = nn.Sequential(
            nn.Linear(adjusted_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.3),

            nn.Linear(hidden, hidden),   
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.3),

            nn.Linear(hidden, num_classes)
        )

        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, x, features):
        z_img = self.backbone(x)                   # [B, 768]
        z = torch.cat((z_img, features), dim=1)    # [B, 768 + features_dim]
        logits = self.ensemble_model(z)            # [B, num_classes]
        return logits

    def _common_step(self, batch, batch_idx):
        images, features, labels = batch
        logits = self.forward(images, features)
        loss = self.fn_loss(logits, labels)
        preds = torch.argmax(logits, 1)
        return images, features, labels, logits, loss, preds

    def training_step(self, batch, batch_idx):
        images, features, labels, logits, loss, preds = self._common_step(batch, batch_idx)
        self.train_accuracy(preds, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images, features, labels, logits, loss, preds = self._common_step(batch, batch_idx)
        self.val_accuracy(preds, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True, on_epoch=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        images, features, labels, logits, loss, preds = self._common_step(batch, batch_idx)
        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_confusion_matrix(preds, labels)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy.compute(), prog_bar=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True)
        self.log("test_precision", self.test_precision.compute(), prog_bar=True)
        self.log("test_recall", self.test_recall.compute(), prog_bar=True)

        return {
            "test_loss": loss,
            "test_accuracy": self.test_accuracy.compute(),
            "test_f1": self.test_f1.compute(),
            "test_precision": self.test_precision.compute(),
            "test_recall": self.test_recall.compute()
        }

    def on_test_epoch_end(self):
        self.test_accuracy.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        conf_matrix_value = self.test_confusion_matrix.compute().cpu().numpy()
        self.test_confusion_matrix.reset()
        print("✅ Matriz de Confusão calculada após o teste.")
        return conf_matrix_value

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay,
                                      betas=self.optimizer_momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class CustomModelTriple_MLP2(pl.LightningModule):
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
                 layer_scale: float):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.features_dim = features_dim
        self.layer_scale = layer_scale
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_momentum = optimizer_momentum

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

        self.convnext_model = models.convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.DEFAULT,
            drop_path_rate=drop_path_rate
        )
        self.convnext_model.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True)
        )

        self.swint_model = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.swint_model.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True)
        )


        self.image_dim = 1536  

        adjusted_dim = self.image_dim + self.features_dim
        hidden = int(adjusted_dim * layer_scale)

        self.ensemble_model = nn.Sequential(
            nn.Linear(adjusted_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.3),

            nn.Linear(hidden, hidden),           
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.3),

            nn.Linear(hidden, num_classes)
        )

        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, x, features):
        x_conv = self.convnext_model(x)
        x_swin = self.swint_model(x)
        x_img = torch.cat((x_conv, x_swin), dim=1)
        x_total = torch.cat((x_img, features), dim=1)
        return self.ensemble_model(x_total)

    def _common_step(self, batch, batch_idx):
        images, features, labels = batch
        logits = self.forward(images, features)
        loss = self.fn_loss(logits, labels)
        preds = torch.argmax(logits, 1)
        return images, features, labels, logits, loss, preds

    def training_step(self, batch, batch_idx):
        images, features, labels, logits, loss, preds = self._common_step(batch, batch_idx)
        self.train_accuracy(preds, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        images, features, labels, logits, loss, preds = self._common_step(batch, batch_idx)
        self.val_accuracy(preds, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True, on_epoch=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        images, features, labels, logits, loss, preds = self._common_step(batch, batch_idx)
        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_confusion_matrix(preds, labels)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy.compute(), prog_bar=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True)
        self.log("test_precision", self.test_precision.compute(), prog_bar=True)
        self.log("test_recall", self.test_recall.compute(), prog_bar=True)

        return {
            "test_loss": loss,
            "test_accuracy": self.test_accuracy.compute(),
            "test_f1": self.test_f1.compute(),
            "test_precision": self.test_precision.compute(),
            "test_recall": self.test_recall.compute()
        }

    def on_test_epoch_end(self):
        self.test_accuracy.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        conf_matrix_value = self.test_confusion_matrix.compute().cpu().numpy()
        self.test_confusion_matrix.reset()
        print("✅ Matriz de Confusão calculada após o teste.")
        return conf_matrix_value

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay,
                                      betas=self.optimizer_momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class CustomModel_desbalanced(pl.LightningModule):
    def __init__(self, tmodel, name_dataset, shape, epochs, learning_rate,
                 drop_path_rate, num_classes, label_smoothing, optimizer_momentum,
                 weight_decay, layer_scale, class_weights: torch.Tensor | None = None):
        
        super(CustomModel, self).__init__()

        self.save_hyperparameters(ignore=["method", "metric.goal", "metric.name","parameters.batch_size",
                                          "parameters.layer_scale", "parameters.learning_rate.distribution",
                                          "parameters.learning_rate.max", "parameters.learning_rate.min"])
        
        self.tmodel = tmodel
        self.name_dataset = name_dataset
        self.shape = shape
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.optimizer_momentum = optimizer_momentum
        self.weight_decay= weight_decay
        self.layer_scale = layer_scale

        # (novo) pesos por classe opcionais, registrados como buffer para mover com .to(device)
        if class_weights is not None:
            class_weights = class_weights.float()
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self._rebuild_loss()

        self.model_dim = 0
        self.validation_step_outputs = []
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)       
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes) 
        
        self.train_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=num_classes)
        
        self.train_recall = Recall(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)
        self.test_recall = Recall(task="multiclass", num_classes=num_classes)

        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

        # Escolha do modelo
        if tmodel == "convnext_t":
            self.model_dim = 768
            self.dl_model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT, 
                                            drop_path_rate=self.drop_path_rate)
            self.sequential_layers = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(self.model_dim, eps=1e-6, elementwise_affine=True),
            )
            self.dl_model.classifier = self.sequential_layers

        if tmodel == "swint_t":
            self.model_dim = 768
            self.dl_model = swin_t(weights=Swin_T_Weights.DEFAULT)
            self.sequential_layers = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(self.model_dim, eps=1e-6, elementwise_affine=True),
                )
            self.dl_model.head = self.sequential_layers

        # Modelo de combinação ajustado
        adjusted_dim = self.model_dim
        scaled_dim = int(adjusted_dim * self.layer_scale)

        self.model = nn.Sequential(
            nn.Linear(adjusted_dim, scaled_dim),
            nn.GELU(approximate='none'),
            nn.LayerNorm(scaled_dim),
            nn.Dropout(p=0.3),
            nn.Linear(scaled_dim, self.num_classes)
        )

    # (novo) permite atualizar pesos por classe depois que o datamodule estiver pronto
    def set_class_weights(self, class_weights: torch.Tensor | None):
        if class_weights is not None:
            class_weights = class_weights.float().to(self.device)
            self.class_weights = class_weights  # buffer já existente
        else:
            self.class_weights = None
        self._rebuild_loss()

    def _rebuild_loss(self):
        if self.class_weights is not None:
            self.fn_loss = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=self.label_smoothing)
        else:
            self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
    def forward(self, x):
        x = self.dl_model(x)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels, logits, loss, preds = self._commom_step(batch, batch_idx)
        self.train_accuracy(preds, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        images, labels, logits, loss, preds = self._commom_step(batch, batch_idx)
        self.val_accuracy(preds, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True, on_epoch=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        images, labels, logits, loss, preds = self._commom_step(batch, batch_idx)

        # Atualiza as métricas corretamente
        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_confusion_matrix.update(preds, labels)  # (novo) acumula matriz

        # Loga as métricas corretamente
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy.compute(), prog_bar=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True)
        self.log("test_precision", self.test_precision.compute(), prog_bar=True)
        self.log("test_recall", self.test_recall.compute(), prog_bar=True)

        return {
            "test_loss": loss,
            "test_accuracy": self.test_accuracy.compute(),
            "test_f1": self.test_f1.compute(),
            "test_precision": self.test_precision.compute(),
            "test_recall": self.test_recall.compute()        
            }
    
    def on_test_epoch_end(self):
        self.test_accuracy.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()

        conf_matrix_value = self.test_confusion_matrix.compute().cpu().numpy()
        self.test_confusion_matrix.reset()
        print("Matriz de Confusão calculada após o teste.")
        return conf_matrix_value

    def on_validation_epoch_end(self):
        avg_loss = torch.mean(torch.tensor(self.validation_step_outputs)) if len(self.validation_step_outputs) > 0 else torch.tensor(float('nan'))
        self.log('avg_val_loss', avg_loss)
        self.validation_step_outputs.clear()
        
    def _commom_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        preds = torch.argmax(logits, 1)
        return images, labels, logits, loss, preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            betas = self.optimizer_momentum,
            weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }
