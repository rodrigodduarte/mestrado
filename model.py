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
    

class CustomMLPModel(pl.LightningModule):
    def __init__(self, input_dim, epochs, learning_rate, scale_factor, drop_path_rate,
                 num_classes, label_smoothing, optimizer_momentum, hidden_layers=3, hidden_units=128,
                 warmup_steps = 10, weight_decay = 0.005):
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
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

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
            layers.append(nn.Linear(in_features, 4*hidden_units, bias=True)),
            layers.append(nn.LayerNorm(4*hidden_units, eps=1e-06, elementwise_affine=True)),
            layers.append(nn.GELU(approximate='none')),  # Ativação GELU
            layers.append(nn.Dropout(0.1))  # Dropout para evitar overfitting
            layers.append(nn.Linear(4*hidden_units, hidden_units, bias=True))
            layers.append(nn.LayerNorm(hidden_units, eps=1e-06, elementwise_affine=True)),
            layers.append(nn.GELU(approximate='none')),  # Ativação GELU
            layers.append(nn.Dropout(0.1))  # Dropout para evitar overfitting
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
        # Definir o otimizador
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay= self.weight_decay)

        # Função para o warm-up
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return 1.0

        # Warm-up scheduler com LambdaLR
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # Cosine annealing scheduler após o warm-up
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs - self.warmup_steps)

        # Combinação dos schedulers usando SequentialLR
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )

        # Retornar o otimizador e o scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Step the scheduler per batch
                'frequency': 1  # Aplica o scheduler a cada batch
            }
        }
    
class CustomImageCSVModel(pl.LightningModule):
    def __init__(self, features_dim, epochs, learning_rate, scale_factor,
                 drop_path_rate, num_classes, label_smoothing, optimizer_momentum, hidden_dim_mlp):
        
        super(CustomImageCSVModel, self).__init__()

        self.features_dim = features_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.optimizer_momentum = optimizer_momentum
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.hidden_dim_mlp = hidden_dim_mlp
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        # Backbone ConvNeXt
        self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT, 
                                          drop_path_rate=self.drop_path_rate)
        self.backbone = nn.Sequential(*list(self.model.children())[:-1])  # Remove a última camada (classificador)
        
        self.mlp = nn.Sequential(
            nn.Linear(768 + self.features_dim, 2048),  # Concatenando 768 (do ConvNeXt) com o vetor extra
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, num_classes)
        )
        
    def forward(self, x, features):
        # Extrair as características do ConvNeXt
        model_features = self.backbone(x).flatten(1)  # Flatten as saídas do ConvNeXt (não inclui o batch size)
        
        # Concatenar o vetor extra de características
        features = torch.cat((model_features, features), dim=1)  # Concatenar ao longo da dimensão das características
        print(features.shape)
        # Passar pelas camadas finais (MLP)
        out = self.mlp(features)
        return out

    def training_step(self, batch, batch_idx):
        images, features, labels = batch
        logits = self.forward(images, features)
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
        images, features, labels = batch
        logits = self.forward(images, features)
        loss = self.fn_loss(logits, labels)
        preds = torch.argmax(logits, 1)
        
        # Calcular a precisão para validação
        self.val_accuracy(preds, labels)
        
        # Logar a perda e a acurácia no conjunto de validação
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True, on_epoch=True)
        
        # Retornar a perda e a acurácia
        return {'val_loss': loss}
    

    def test_step(self, batch, batch_idx):
        images, features, labels = batch
        logits = self.forward(images, features)
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
    