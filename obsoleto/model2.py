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



class ResidualBlockWithLayerScale(nn.Module):
    def __init__(self, in_features, out_features, layer_scale_init_value=1e-6, dropout_prob=0.3):
        super(ResidualBlockWithLayerScale, self).__init__()
        self.layer_norm1 = nn.LayerNorm(in_features)  # A LayerNorm precisa ser do tamanho de in_features
        self.linear1 = nn.Linear(in_features, out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.layer_norm2 = nn.LayerNorm(out_features)  # E esta LayerNorm precisa ser do tamanho de out_features
        self.linear2 = nn.Linear(out_features, in_features)

        # Inicializa os pesos do LayerScale com um valor pequeno
        self.layer_scale1 = nn.Parameter(layer_scale_init_value * torch.ones(out_features), requires_grad=True)
        self.layer_scale2 = nn.Parameter(layer_scale_init_value * torch.ones(in_features), requires_grad=True)

    def forward(self, x):
        # Caminho residual 1
        residual = x
        x = self.layer_norm1(x)  # Normaliza o tensor de entrada (com base no in_features)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Aplicar LayerScale e adicionar o resíduo
        x = residual + x * self.layer_scale1

        # Caminho residual 2
        residual = x
        x = self.layer_norm2(x)  # Normaliza o tensor de entrada (com base no out_features)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Aplicar LayerScale e adicionar o resíduo
        return residual + x * self.layer_scale2


class CustomEnsembleModel(pl.LightningModule):
    def __init__(self, name_dataset, shape, epochs, learning_rate, features_dim, scale_factor,
                 drop_path_rate, num_classes, label_smoothing, optimizer_momentum,
                 weight_decay, layer_scale, mlp_vector_model_scale):
        
        super(CustomEnsembleModel, self).__init__()

        self.save_hyperparameters(ignore=["method", "metric.goal", "metric.name","parameters.batch_size",
                                          "parameters.layer_scale", "parameters.learning_rate.distribution",
                                          "parameters.learning_rate.max", "parameters.learning_rate.min"])

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
        self.weight_decay = weight_decay
        self.layer_scale = layer_scale
        self.mlp_vector_model_scale = mlp_vector_model_scale
        self.fn_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        # Modelo ConvNeXt ajustado
        self.dl_model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT, 
                                             drop_path_rate=self.drop_path_rate)
        self.sequential_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(768, eps=1e-6, elementwise_affine=True)  # 768 é o número de características
        )
        self.dl_model.classifier = self.sequential_layers

        # Modelo MLP com blocos residuais e LayerScale
        self.mlp_vector_model = nn.Sequential(
            ResidualBlockWithLayerScale(features_dim, int(self.mlp_vector_model_scale * features_dim),
                                        layer_scale_init_value=self.layer_scale),
            ResidualBlockWithLayerScale(int(self.mlp_vector_model_scale * features_dim),
                                        int(self.mlp_vector_model_scale * features_dim),
                                        layer_scale_init_value=self.layer_scale)
        )

        # Modelo de combinação ajustado com bloco residual
        adjusted_dim = int(self.mlp_vector_model_scale * features_dim) + 768
        scaled_dim = int(adjusted_dim * self.layer_scale)

        self.ensemble_model = nn.Sequential(
            ResidualBlockWithLayerScale(adjusted_dim, scaled_dim, layer_scale_init_value=self.layer_scale),
            nn.Linear(scaled_dim, self.num_classes)
        )

    def forward(self, x, features):
        x = self.dl_model(x)
        features = self.mlp_vector_model(features)

        x = torch.cat((x, features), dim=1)
        x = self.ensemble_model(x)

        return x
