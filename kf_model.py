import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision.models.convnext import (convnext_tiny, ConvNeXt_Tiny_Weights,
                                         convnext_small, ConvNeXt_Small_Weights,
                                         convnext_base, ConvNeXt_Base_Weights)
from torchvision.models.swin_transformer import (swin_t, Swin_T_Weights)
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


class CustomEnsembleModel(nn.Module):
    def __init__(self, tmodel, num_classes, features_dim, mlp_vector_model_scale, layer_scale):
        super(CustomEnsembleModel, self).__init__()
        
        self.tmodel = tmodel
        self.num_classes = num_classes
        self.features_dim = features_dim
        self.mlp_vector_model_scale = mlp_vector_model_scale
        self.layer_scale = layer_scale
        
        if tmodel == "convnext_t":
            self.model_dim = 768
            self.dl_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            self.dl_model.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(self.model_dim),
            )
        elif tmodel == "convnext_s":
            self.model_dim = 768
            self.dl_model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
            self.dl_model.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(self.model_dim),
            )
        elif tmodel == "convnext_b":
            self.model_dim = 1024
            self.dl_model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
            self.dl_model.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(self.model_dim),
            )
        elif tmodel == "swint_t":
            self.model_dim = 768
            self.dl_model = swin_t(weights=Swin_T_Weights.DEFAULT)
            self.dl_model.head = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.LayerNorm(self.model_dim),
            )
        
        self.mlp_vector_model = nn.Sequential(
            nn.Linear(features_dim, int(mlp_vector_model_scale * features_dim)),
            nn.GELU(),
            nn.LayerNorm(int(mlp_vector_model_scale * features_dim)),
            nn.Dropout(p=0.3)
        )
        
        adjusted_dim = int(mlp_vector_model_scale * features_dim) + self.model_dim
        scaled_dim = int(adjusted_dim * layer_scale)

        self.ensemble_model = nn.Sequential(
            nn.Linear(adjusted_dim, scaled_dim),
            nn.GELU(),
            nn.LayerNorm(scaled_dim),
            nn.Dropout(p=0.3),
            nn.Linear(scaled_dim, num_classes)
        )
    
    def forward(self, x, features):
        x = self.dl_model(x)
        features = self.mlp_vector_model(features)
        x = torch.cat((x, features), dim=1)
        x = self.ensemble_model(x)
        return x