import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights

class CustomModelTriple(pl.LightningModule):
    def __init__(self,
                 name_dataset: str,
                 shape: tuple,
                 epochs: int,
                 learning_rate: float,
                 features_dim: int,
                 scale_factor: float,
                 drop_path_rate: float,
                 num_classes: int,
                 label_smoothing: float,
                 optimizer_momentum: tuple,
                 weight_decay: float,
                 layer_scale: float):
        super().__init__()
        
        # armazenar hiperparâmetros para facilitar o load_from_checkpoint
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.features_dim = features_dim
        self.layer_scale = layer_scale
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_momentum = optimizer_momentum

        # === Backbone ConvNeXt Tiny ===
        self.convnext_model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT,
                                                   drop_path_rate=drop_path_rate)
        self.convnext_model.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True)
        )

        # === Backbone Swin Transformer Tiny ===
        self.swint_model = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.swint_model.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True)
        )

        # saída combinada dos modelos de imagem
        self.image_dim = 1536

        # camada final recebe (convnext + swin + vetor de características)
        adjusted_dim = self.image_dim + self.features_dim
        scaled_dim = int(adjusted_dim * layer_scale)
        
        self.ensemble_model = nn.Sequential(
            nn.Linear(adjusted_dim, scaled_dim),
            nn.GELU(),
            nn.LayerNorm(scaled_dim),
            nn.Dropout(0.3),
            nn.Linear(scaled_dim, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, x, features):
        x_conv = self.convnext_model(x)
        x_swin = self.swint_model(x)
        x_img = torch.cat((x_conv, x_swin), dim=1)
        x_total = torch.cat((x_img, features), dim=1)
        return self.ensemble_model(x_total)

    def training_step(self, batch, batch_idx):
        images, features, labels = batch
        outputs = self(images, features)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, features, labels = batch
        outputs = self(images, features)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, features, labels = batch
        outputs = self(images, features)
        loss = self.criterion(outputs, labels)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay,
                                      betas=self.optimizer_momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
