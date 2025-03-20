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
    def __init__(self, tmodel, name_dataset, epochs, shape, learning_rate, scale_factor,
                 drop_path_rate, num_classes, label_smoothing, optimizer_momentum):
        
        super(CustomModel, self).__init__()

        self.save_hyperparameters()

        self.tmodel = tmodel
        self.name_dataset = name_dataset
        self.epochs = epochs
        self.shape = shape
        self.learning_rate = learning_rate
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.optimizer_momentum = optimizer_momentum
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.train_precision = Precision(task="multiclass", num_classes=num_classes)
        self.train_recall = Recall(task="multiclass", num_classes=num_classes)

        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_precision = Precision(task="multiclass", num_classes=num_classes)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes)

        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=num_classes)
        self.test_recall = Recall(task="multiclass", num_classes=num_classes)

        

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
        if tmodel == "swint_t":
            self.model_dim = 768
            self.model = swin_t(weights=Swin_T_Weights.DEFAULT)
            self.sequential_layers = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(self.model_dim, eps=1e-6, elementwise_affine=True),
                )
            self.model.head = self.sequential_layers
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
        
        # Concatena todas as previsões e rótulos coletados durante o teste
        all_preds = torch.cat(self.test_preds)
        all_labels = torch.cat(self.test_labels)

        # Define a matriz de confusão se necessário
        conf_matrix_metric = MulticlassConfusionMatrix(num_classes=self.hparams.num_classes)
        conf_matrix_value = conf_matrix_metric(all_preds, all_labels).cpu().numpy()

        print("✅ Matriz de Confusão calculada após o teste.")
        
        # Limpar listas para evitar acúmulo de memória em execuções futuras
        self.test_preds.clear()
        self.test_labels.clear()

        return conf_matrix_value


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
      

class CustomEnsembleModel(pl.LightningModule):
    def __init__(self, tmodel, name_dataset, shape, epochs, learning_rate, features_dim, scale_factor,
                 drop_path_rate, num_classes, label_smoothing, optimizer_momentum,
                 weight_decay, layer_scale, mlp_vector_model_scale):
        
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
        self.scale_factor = scale_factor
        self.drop_path_rate = drop_path_rate
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.optimizer_momentum = optimizer_momentum
        self.weight_decay= weight_decay
        self.layer_scale = layer_scale
        self.mlp_vector_model_scale = mlp_vector_model_scale
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


        # Modelo MLP ajustado
        self.mlp_vector_model = nn.Sequential(
            nn.Linear(features_dim, int((self.mlp_vector_model_scale) * features_dim)),
            nn.GELU(approximate='none'),
            nn.LayerNorm(int((self.mlp_vector_model_scale) * features_dim)),
            nn.Dropout(p=0.3)
        )

        # Modelo de combinação ajustado
        adjusted_dim = int((self.mlp_vector_model_scale) * features_dim) + self.model_dim
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
        features = self.mlp_vector_model(features)

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

