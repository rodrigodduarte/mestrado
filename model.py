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

class CustomConvNeXtTiny(pl.LightningModule):
    def __init__(self, epochs,learning_rate, scale_factor,
                 drop_path_rate, num_classes, label_smoothing):
        
        super(CustomConvNeXtTiny, self).__init__()
        
        # Carregar o modelo ConvNeXt-Tiny com pesos pré-treinados
        self.model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT, 
                                          drop_path_rate=drop_path_rate)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.label_smoothing = label_smoothing
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Definir as camadas sequenciais para classificação
        self.sequential_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True),
            nn.Linear(in_features=768, out_features=num_classes, bias=True)
        )
        
        # Substituir a camada classifier do modelo
        self.model.classifier = self.sequential_layers
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)


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
        # Prepare the list for optimizer param groups
        param_groups = []

        # Scale learning rate for each block in the model's features
        for i, block in enumerate(self.model.features):
            lr = self.learning_rate * (self.scale_factor ** i)
            param_groups.append({'params': block.parameters(), 'lr': lr})

        # Add classifier parameters with final scaled learning rate
        classifier_lr = self.learning_rate * (self.scale_factor ** len(self.model.features))
        param_groups.append({'params': self.model.classifier.parameters(), 'lr': classifier_lr})

        # Define the optimizer
        optimizer = torch.optim.Adam(param_groups)

        # Define scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)  # Example with T_max=10

        # Return both optimizer and scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Step the scheduler per epoch
                'monitor': 'val_loss',  # Optional, monitor val_loss (useful for other schedulers)
                'frequency': 1,  # Apply the scheduler every epoch
            }
        }


class CustomConvNeXtSmall(pl.LightningModule):
    def __init__(self, epochs,learning_rate, scale_factor,
                 drop_path_rate, num_classes, label_smoothing):
        
        super(CustomConvNeXtSmall, self).__init__()
        
        # Carregar o modelo ConvNeXt-Tiny com pesos pré-treinados
        self.model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT, 
                                          drop_path_rate=drop_path_rate)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.label_smoothing = label_smoothing
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Definir as camadas sequenciais para classificação
        self.sequential_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True),
            nn.Linear(in_features=768, out_features=num_classes, bias=True)
        )
        
        # Substituir a camada classifier do modelo
        self.model.classifier = self.sequential_layers
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)

        # Calcular a precisão
        self.train_accuracy(logits, labels)
        
        # Logar a perda e a acurácia
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        # Retornar a perda para o processamento posterior
        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        
        # Calcular a precisão para validação
        self.val_accuracy(logits, labels)
        
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

        # Calcular a precisão
        self.test_accuracy(logits, labels)
        self.log('test_accuracy', self.test_accuracy)


    def configure_optimizers(self):
        # Prepare the list for optimizer param groups
        param_groups = []

        # Scale learning rate for each block in the model's features
        for i, block in enumerate(self.model.features):
            lr = self.learning_rate * (self.scale_factor ** i)
            param_groups.append({'params': block.parameters(), 'lr': lr})

        # Add classifier parameters with final scaled learning rate
        classifier_lr = self.learning_rate * (self.scale_factor ** len(self.model.features))
        param_groups.append({'params': self.model.classifier.parameters(), 'lr': classifier_lr})

        # Define the optimizer with parameter groups
        optimizer = torch.optim.Adam(param_groups)

        # Define scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)  # Example with T_max=10

        # Return both optimizer and scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Step the scheduler per epoch
                'monitor': 'val_loss',  # Optional, monitor val_loss (useful for other schedulers)
                'frequency': 1,  # Apply the scheduler every epoch
            }
        }


class CustomConvNeXtBase(pl.LightningModule):
    def __init__(self, epochs,learning_rate, scale_factor,
                 drop_path_rate, num_classes, label_smoothing):
        
        super(CustomConvNeXtBase, self).__init__()
        
        # Carregar o modelo ConvNeXt-Tiny com pesos pré-treinados
        self.model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT, 
                                          drop_path_rate=drop_path_rate)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.label_smoothing = label_smoothing
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Definir as camadas sequenciais para classificação
        self.sequential_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(1024, eps=1e-6, elementwise_affine=True),
            nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        )
        
        # Substituir a camada classifier do modelo
        self.model.classifier = self.sequential_layers
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)

        # Calcular a precisão
        self.train_accuracy(logits, labels)
        
        # Logar a perda e a acurácia
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        # Retornar a perda para o processamento posterior
        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        
        # Calcular a precisão para validação
        self.val_accuracy(logits, labels)
        
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

        # Calcular a precisão
        self.test_accuracy(logits, labels)
        self.log('test_accuracy', self.test_accuracy)


    def configure_optimizers(self):
        # Prepare the list for optimizer param groups
        param_groups = []

        # Scale learning rate for each block in the model's features
        for i, block in enumerate(self.model.features):
            lr = self.learning_rate * (self.scale_factor ** i)
            param_groups.append({'params': block.parameters(), 'lr': lr})

        # Add classifier parameters with final scaled learning rate
        classifier_lr = self.learning_rate * (self.scale_factor ** len(self.model.features))
        param_groups.append({'params': self.model.classifier.parameters(), 'lr': classifier_lr})

        # Define the optimizer with parameter groups
        optimizer = torch.optim.Adam(param_groups)

        # Define scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)  # Example with T_max=10

        # Return both optimizer and scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Step the scheduler per epoch
                'monitor': 'val_loss',  # Optional, monitor val_loss (useful for other schedulers)
                'frequency': 1,  # Apply the scheduler every epoch
            }
        }


class CustomConvNeXtLarge(pl.LightningModule):
    def __init__(self, epochs,learning_rate, scale_factor,
                 drop_path_rate, num_classes, label_smoothing):
        
        super(CustomConvNeXtLarge, self).__init__()
        
        # Carregar o modelo ConvNeXt-Tiny com pesos pré-treinados
        self.model = models.convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT, 
                                          drop_path_rate=drop_path_rate)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.label_smoothing = label_smoothing
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Definir as camadas sequenciais para classificação
        self.sequential_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(1536, eps=1e-6, elementwise_affine=True),
            nn.Linear(in_features=1536, out_features=num_classes, bias=True)
        )
        
        # Substituir a camada classifier do modelo
        self.model.classifier = self.sequential_layers
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)

        # Calcular a precisão
        self.train_accuracy(logits, labels)
        
        # Logar a perda e a acurácia
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        # Retornar a perda para o processamento posterior
        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        
        # Calcular a precisão para validação
        self.val_accuracy(logits, labels)
        
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

        # Calcular a precisão
        self.test_accuracy(logits, labels)
        self.log('test_accuracy', self.test_accuracy)


    def configure_optimizers(self):
        # Prepare the list for optimizer param groups
        param_groups = []

        # Scale learning rate for each block in the model's features
        for i, block in enumerate(self.model.features):
            lr = self.learning_rate * (self.scale_factor ** i)
            param_groups.append({'params': block.parameters(), 'lr': lr})

        # Add classifier parameters with final scaled learning rate
        classifier_lr = self.learning_rate * (self.scale_factor ** len(self.model.features))
        param_groups.append({'params': self.model.classifier.parameters(), 'lr': classifier_lr})

        # Define the optimizer with parameter groups
        optimizer = torch.optim.Adam(param_groups)

        # Define scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)  # Example with T_max=10

        # Return both optimizer and scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Step the scheduler per epoch
                'monitor': 'val_loss',  # Optional, monitor val_loss (useful for other schedulers)
                'frequency': 1,  # Apply the scheduler every epoch
            }
        }
    
    
class CustomCCT_14_7x2_224(pl.LightningModule):
    def __init__(self, epochs,learning_rate, scale_factor,
                 drop_path_rate, num_classes, label_smoothing):
        
        super(CustomCCT_14_7x2_224, self).__init__()
        
        # Carregar o modelo ConvNeXt-Tiny com pesos pré-treinados
        self.model = cct_14_7x2_224(pretrained=True, progress=True)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.label_smoothing = label_smoothing
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Definir as camadas sequenciais para classificação
        self.sequential_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(384, eps=1e-6, elementwise_affine=True),
            nn.Linear(in_features=384, out_features=num_classes, bias=True)
        )
        
        # Substituir a camada classifier do modelo
        self.model.classifier.fc = self.sequential_layers
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)

        # Calcular a precisão
        self.train_accuracy(logits, labels)
        
        # Logar a perda e a acurácia
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        # Retornar a perda para o processamento posterior
        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        
        # Calcular a precisão para validação
        self.val_accuracy(logits, labels)
        
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

        # Calcular a precisão
        self.test_accuracy(logits, labels)
        self.log('test_accuracy', self.test_accuracy)


    def configure_optimizers(self):
        # Definir o otimizador com os grupos de parâmetros
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Definir o scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Retornar o otimizador e o scheduler
        return [optimizer], [scheduler]


class CustomCCT_14_7x2_384(pl.LightningModule):
    def __init__(self, epochs,learning_rate, scale_factor,
                 drop_path_rate, num_classes, label_smoothing):
        
        super(CustomCCT_14_7x2_384, self).__init__()
        
        # Carregar o modelo ConvNeXt-Tiny com pesos pré-treinados
        self.model = cct_14_7x2_384(pretrained=True, progress=True)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.label_smoothing = label_smoothing
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Definir as camadas sequenciais para classificação
        self.sequential_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(384, eps=1e-6, elementwise_affine=True),
            nn.Linear(in_features=384, out_features=num_classes, bias=True)
        )
        
        # Substituir a camada classifier do modelo
        self.model.classifier.fc = self.sequential_layers
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)

        # Calcular a precisão
        self.train_accuracy(logits, labels)
        
        # Logar a perda e a acurácia
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        # Retornar a perda para o processamento posterior
        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        
        # Calcular a precisão para validação
        self.val_accuracy(logits, labels)
        
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

        # Calcular a precisão
        self.test_accuracy(logits, labels)
        self.log('test_accuracy', self.test_accuracy)


    def configure_optimizers(self):
        # Definir o otimizador com os grupos de parâmetros
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Definir o scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Retornar o otimizador e o scheduler
        return [optimizer], [scheduler]


class CustomCCT_14_7x2_384_fl(pl.LightningModule):
    def __init__(self, epochs,learning_rate, scale_factor,
                 drop_path_rate, num_classes, label_smoothing):
        
        super(CustomCCT_14_7x2_384_fl, self).__init__()
        
        # Carregar o modelo ConvNeXt-Tiny com pesos pré-treinados
        self.model = cct_14_7x2_384_fl(pretrained=True, progress=True)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.label_smoothing = label_smoothing
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Definir as camadas sequenciais para classificação
        self.sequential_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(384, eps=1e-6, elementwise_affine=True),
            nn.Linear(in_features=384, out_features=num_classes, bias=True)
        )
        
        # Substituir a camada classifier do modelo
        self.model.classifier.fc = self.sequential_layers
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)

        # Calcular a precisão
        self.train_accuracy(logits, labels)
        
        # Logar a perda e a acurácia
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        # Retornar a perda para o processamento posterior
        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.fn_loss(logits, labels)
        
        # Calcular a precisão para validação
        self.val_accuracy(logits, labels)
        
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

        # Calcular a precisão
        self.test_accuracy(logits, labels)
        self.log('test_accuracy', self.test_accuracy)


    def configure_optimizers(self):
        # Definir o otimizador com os grupos de parâmetros
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Definir o scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Retornar o otimizador e o scheduler
        return [optimizer], [scheduler]


class CustomSwinTransformerTiny(pl.LightningModule):
    def __init__(self, epochs,learning_rate, scale_factor,
                 drop_path_rate, num_classes, label_smoothing):
        
        super(CustomSwinTransformerTiny, self).__init__()
        
        # Carregar o modelo ConvNeXt-Tiny com pesos pré-treinados
        self.model = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.label_smoothing = label_smoothing
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Substituir a camada classifier do modelo
        self.model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)


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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Definir o scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Retornar o otimizador e o scheduler
        return [optimizer], [scheduler]


class CustomSwinTransformerSmall(pl.LightningModule):
    def __init__(self, epochs,learning_rate, scale_factor,
                 drop_path_rate, num_classes, label_smoothing):
        
        super(CustomSwinTransformerSmall, self).__init__()
        
        self.model = swin_s(weights=Swin_S_Weights.DEFAULT)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.label_smoothing = label_smoothing
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        self.model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)


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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Definir o scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Retornar o otimizador e o scheduler
        return [optimizer], [scheduler]


class CustomSwinTransformerBase(pl.LightningModule):
    def __init__(self, epochs,learning_rate, scale_factor,
                 drop_path_rate, num_classes, label_smoothing):
        
        super(CustomSwinTransformerBase, self).__init__()
        
        self.model = swin_b(weights=Swin_B_Weights.DEFAULT)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.label_smoothing = label_smoothing
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        self.model.head = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)


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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Definir o scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Retornar o otimizador e o scheduler
        return [optimizer], [scheduler]