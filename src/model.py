"""
============================================================
DermaScope AI — Modèles Multimodaux FiLM
============================================================
Architecture avancée utilisant la Feature-wise Linear Modulation (FiLM)
pour conditionner l'extraction de caractéristiques visuelles par les
métadonnées cliniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_b4, EfficientNet_B4_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights
)

__all__ = ['FiLM_Layer', 'Dermascope_FiLM_EfficientNet', 'Dermascope_FiLM_Alternative', 'DermascopeFocalLoss']

class FiLM_Layer(nn.Module):
    """
    Couche de Feature-wise Linear Modulation.
    Les métadonnées génèrent les paramètres Gamma et Beta qui vont
    multiplier et additionner les features visuelles.
    """
    def __init__(self, tabular_dim: int, vision_dim: int):
        super().__init__()
        self.gamma = nn.Linear(tabular_dim, vision_dim)
        self.beta = nn.Linear(tabular_dim, vision_dim)
        
    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Modulation affine : Vision * (1 + Gamma(Tabular)) + Beta(Tabular)
        return v * (1.0 + self.gamma(t)) + self.beta(t)


class Dermascope_FiLM_EfficientNet(nn.Module):
    def __init__(self, num_tabular_features: int):
        super().__init__()
        self.vision = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        num_v = self.vision.classifier[1].in_features
        self.vision.classifier = nn.Identity()
        
        self.compress = nn.Sequential(
            nn.Linear(num_v, 512), 
            nn.BatchNorm1d(512), 
            nn.SiLU()
        )
        
        self.tabular = nn.Sequential(
            nn.Linear(num_tabular_features, 64), 
            nn.BatchNorm1d(64), 
            nn.SiLU(), 
            nn.Dropout(0.2), 
            nn.Linear(64, 32), 
            nn.BatchNorm1d(32), 
            nn.SiLU()
        )
        
        self.film = FiLM_Layer(tabular_dim=32, vision_dim=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256), 
            nn.SiLU(), 
            nn.Dropout(0.4), 
            nn.Linear(256, 1)
        )
        
    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        v = self.compress(self.vision(images))
        t = self.tabular(metadata)
        fused = self.film(v, t)
        return self.classifier(fused)


class Dermascope_FiLM_Alternative(nn.Module):
    def __init__(self, num_tabular_features: int, modele: str = "resnet50"):
        super().__init__()
        if modele == "resnet50": 
            self.vision = resnet50(weights=ResNet50_Weights.DEFAULT)
            num_v = self.vision.fc.in_features
            self.vision.fc = nn.Identity()
        elif modele == "densenet121": 
            self.vision = densenet121(weights=DenseNet121_Weights.DEFAULT)
            num_v = self.vision.classifier.in_features
            self.vision.classifier = nn.Identity()
        else:
            raise ValueError("Modèle non supporté. Choisissez resnet50 ou densenet121.")
            
        self.compress = nn.Sequential(
            nn.Linear(num_v, 512), 
            nn.BatchNorm1d(512), 
            nn.SiLU()
        )
        
        self.tabular = nn.Sequential(
            nn.Linear(num_tabular_features, 64), 
            nn.BatchNorm1d(64), 
            nn.SiLU(), 
            nn.Dropout(0.2), 
            nn.Linear(64, 32), 
            nn.BatchNorm1d(32), 
            nn.SiLU()
        )
        
        self.film = FiLM_Layer(tabular_dim=32, vision_dim=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256), 
            nn.SiLU(), 
            nn.Dropout(0.4), 
            nn.Linear(256, 1)
        )
        
    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        v = self.compress(self.vision(images))
        t = self.tabular(metadata)
        fused = self.film(v, t)
        return self.classifier(fused)


class DermascopeFocalLoss(nn.Module):
    """
    Focal Loss optimisée pour les datasets déséquilibrés.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0): 
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()
