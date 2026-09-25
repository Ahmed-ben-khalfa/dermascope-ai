"""
Dermascope AI — FiLM Model Architectures
Feature-wise Linear Modulation for multimodal melanoma detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_b4, EfficientNet_B4_Weights,
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights
)

__all__ = [
    'FiLM_Layer',
    'Dermascope_FiLM_EfficientNet',
    'Dermascope_FiLM_Alternative',
    'DermascopeFocalLoss'
]


class FiLM_Layer(nn.Module):
    """Feature-wise Linear Modulation: output = v * (1 + gamma(t)) + beta(t)"""

    def __init__(self, tabular_dim: int, vision_dim: int):
        super().__init__()
        self.gamma = nn.Linear(tabular_dim, vision_dim)
        self.beta = nn.Linear(tabular_dim, vision_dim)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return v * (1.0 + self.gamma(t)) + self.beta(t)


class Dermascope_FiLM_EfficientNet(nn.Module):
    """EfficientNet-B4 backbone with FiLM conditioning."""

    def __init__(self, num_tabular_features: int):
        super().__init__()
        self.vision = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        num_v = self.vision.classifier[1].in_features
        self.vision.classifier = nn.Identity()

        self.compress = nn.Sequential(
            nn.Linear(num_v, 512), nn.BatchNorm1d(512), nn.SiLU()
        )
        self.tabular = nn.Sequential(
            nn.Linear(num_tabular_features, 64), nn.BatchNorm1d(64), nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU()
        )
        self.film = FiLM_Layer(tabular_dim=32, vision_dim=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, images, metadata):
        v = self.compress(self.vision(images))
        t = self.tabular(metadata)
        return self.classifier(self.film(v, t))


class Dermascope_FiLM_Alternative(nn.Module):
    """ResNet-50 or DenseNet-121 backbone with FiLM conditioning."""

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
            raise ValueError(f"Unknown backbone: {modele}")

        self.compress = nn.Sequential(
            nn.Linear(num_v, 512), nn.BatchNorm1d(512), nn.SiLU()
        )
        self.tabular = nn.Sequential(
            nn.Linear(num_tabular_features, 64), nn.BatchNorm1d(64), nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.SiLU()
        )
        self.film = FiLM_Layer(tabular_dim=32, vision_dim=512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, images, metadata):
        v = self.compress(self.vision(images))
        t = self.tabular(metadata)
        return self.classifier(self.film(v, t))


class DermascopeFocalLoss(nn.Module):
    """Focal Loss for imbalanced binary classification."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (alpha_t * ((1 - pt) ** self.gamma) * bce).mean()
