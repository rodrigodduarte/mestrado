import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

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
from torchmetrics import Accuracy

import pytorch_lightning as pl

import config
from compact_transform.src import cct_14_7x2_224, cct_14_7x2_384, cct_14_7x2_384_fl


class CustomModel(pl.LightningModule):
    def __init__(self, tmodel, epochs, learning_rate, scale_factor,
                 drop_path_rate, num_classes, label_smoothing, optimizer_momentum):
        
        super(CustomModel, self).__init__()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.optimizer_momentum = optimizer_momentum
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        # Escolha do modelo
        if tmodel == "convnext_t":
            self.model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT, 
                                            drop_path_rate=self.drop_path_rate)
            self.sequential_layers = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(768, eps=1e-6, elementwise_affine=True),
                nn.Linear(in_features=768, out_features=self.num_classes, bias=True)
            )
            self.model.classifier = self.sequential_layers
        
        if tmodel == "convnext_s":
            self.model = models.convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT, 
                                            drop_path_rate=self.drop_path_rate)
            self.sequential_layers = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(768, eps=1e-6, elementwise_affine=True),
                nn.Linear(in_features=768, out_features=self.num_classes, bias=True)
            )
            self.model.classifier = self.sequential_layers

        if tmodel == "convnext_b":
            self.model = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT,
                                              drop_path_rate = self.drop_path_rate)
            self.sequential_layers = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(1024, eps=1e-6, elementwise_affine=True),
                nn.Linear(in_features=1024, out_features=self.num_classes, bias=True)
            )
            self.model.classifier = self.sequential_layers

        if tmodel == "convnext_l":
            self.model = models.convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT, 
                                    drop_path_rate=self.drop_path_rate)
            self.sequential_layers = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(1536, eps=1e-6, elementwise_affine=True),
                nn.Linear(in_features=1536, out_features=self.num_classes, bias=True)
            )
            self.model.classifier = self.sequential_layers

        if tmodel == "swint_t":
            self.model = swin_t(weights=Swin_T_Weights.DEFAULT)
            self.model.head = nn.Linear(in_features=768, out_features=self.num_classes, bias=True)

        if tmodel == "swint_s":
            self.model = swin_s(weights=Swin_S_Weights.DEFAULT)
            self.model.head = nn.Linear(in_features=768, out_features=self.num_classes, bias=True)

        if tmodel == "swint_b":
            self.model = swin_b(weights = Swin_B_Weights.DEFAULT)
            self.model.head = nn.Linear(in_features=1024, out_features=self.num_classes, bias=True)
        
        if tmodel == "cct_224":
            self.model = cct_14_7x2_224(pretrained=True, progress=True)
            self.sequential_layers = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(384, eps=1e-6, elementwise_affine=True),
                nn.Linear(in_features=384, out_features=self.num_classes, bias=True)
            )
            self.model.classifier.fc = self.sequential_layers

        if tmodel == "cct_384":
            self.model = cct_14_7x2_384(pretrained=True, progress=True)
            self.sequential_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(384, eps=1e-6, elementwise_affine=True),
            nn.Linear(in_features=384, out_features=self.num_classes, bias=True)
        )
            self.model.classifier.fc = self.sequential_layers

        if tmodel == "cct_384_fl":
            self.model = cct_14_7x2_384_fl(pretrained=True, progress=True)
            self.sequential_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(384, eps=1e-6, elementwise_affine=True),
            nn.Linear(in_features=384, out_features=self.num_classes, bias=True)
        )
            self.model.classifier.fc = self.sequential_layers


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        preds = torch.argmax(logits, 1)

        # Calcular a precisão
        self.train_accuracy(preds, labels)
        
        # Logar a perda e a acurácia
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        # Retornar a perda para o processamento posterior
        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        preds = torch.argmax(logits, 1)
        
        # Calcular a precisão para validação
        self.val_accuracy(preds, labels)
        
        # Logar a perda e a acurácia no conjunto de validação
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True, on_epoch=True)
        
        # Retornar a perda e a acurácia
        return {'val_loss': loss}
    
    # def on_train_epoch_end(self):
    #     # Acessar a perda média do treino automaticamente através do logger
    #     avg_loss = self.trainer.callback_metrics['train_loss']

    #     # Imprimir a perda média de treino
    #     print(f'Loss médio do treino na época: {avg_loss:.4f}')
    
    # def on_validation_epoch_end(self):
    #     # Acessar a perda média da validação automaticamente através do logger
    #     avg_val_loss = self.trainer.callback_metrics['val_loss']

    #     # Imprimir a perda média da validação
    #     print(f'Loss médio da validação na época: {avg_val_loss:.4f}')


    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        preds = torch.argmax(logits, 1)

        # Calcular a precisão
        self.test_accuracy(preds, labels)   
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_accuracy, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        # Definir o otimizador com os grupos de parâmetros
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas = self.optimizer_momentum)

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
    

class CustomModel(pl.LightningModule):
    def __init__(self, input_dim, epochs, learning_rate, scale_factor, drop_path_rate,
                 num_classes, label_smoothing, optimizer_momentum, hidden_layers=3, hidden_units=128):
        """
        Inicializa o modelo MLP para classificação usando nn.ModuleList com PyTorch Lightning.

        Args:
            input_dim (int): Número de características (atributos) nos dados de entrada.
            num_classes (int): Número de classes para a classificação.
            hidden_layers (int): Número de camadas ocultas. Padrão é 3.
            hidden_units (int): Número de unidades em cada camada oculta. Padrão é 128.
        """
        super(CustomModel, self).__init__()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.optimizer_momentum = optimizer_momentum
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        # Verificando se 'input_dim' e 'hidden_units' são inteiros
        if not isinstance(input_dim, int) or not isinstance(hidden_units, int):
            raise ValueError("Ambos 'input_dim' e 'hidden_units' devem ser inteiros.")
        
        # Lista de camadas
        layers = []
        in_features = input_dim
        
        # Adicionando as camadas ocultas
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())  # Ativação ReLU
            layers.append(nn.Dropout(0.2))  # Dropout para evitar overfitting
            in_features = hidden_units
        
        # Camada final de classificação
        layers.append(nn.Linear(in_features, num_classes))

        # Usando nn.Sequential para armazenar as camadas
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        # Passando a entrada pelo modelo (nn.Sequential)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        preds = torch.argmax(logits, 1)

        # Calcular a precisão
        self.train_accuracy(preds, labels)
        
        # Logar a perda e a acurácia
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        # Retornar a perda para o processamento posterior
        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        preds = torch.argmax(logits, 1)
        
        # Calcular a precisão para validação
        self.val_accuracy(preds, labels)
        
        # Logar a perda e a acurácia no conjunto de validação
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True, on_epoch=True)
        
        # Retornar a perda e a acurácia
        return {'val_loss': loss}
    
    # def on_train_epoch_end(self):
    #     # Acessar a perda média do treino automaticamente através do logger
    #     avg_loss = self.trainer.callback_metrics['train_loss']

    #     # Imprimir a perda média de treino
    #     print(f'Loss médio do treino na época: {avg_loss:.4f}')
    
    # def on_validation_epoch_end(self):
    #     # Acessar a perda média da validação automaticamente através do logger
    #     avg_val_loss = self.trainer.callback_metrics['val_loss']

    #     # Imprimir a perda média da validação
    #     print(f'Loss médio da validação na época: {avg_val_loss:.4f}')


    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        preds = torch.argmax(logits, 1)

        # Calcular a precisão
        self.test_accuracy(preds, labels)   
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_accuracy, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        # Definir o otimizador com os grupos de parâmetros
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas = self.optimizer_momentum)

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
    