"""
============================================================
DermaScope AI — Modèle et Fonction de Coût
============================================================
Auteur  : Bilel Kahma
Projet  : Détection de Pathologies Cutanées par Deep Learning
============================================================
Architecture du réseau multimodal EfficientNet-B4 + MLP Tabulaire
et implémentation de la Focal Loss binaire.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from src import config

__all__ = ['DermascopeMultimodal', 'DermascopeFocalLoss', 'create_model']

class DermascopeMultimodal(nn.Module):
    """
    Architecture multimodale pour la détection de mélanomes.
    
    Rationnel: Fusion "Late Fusion" des caractéristiques visuelles (EfficientNet-B4)
    et cliniques (MLP) au sein d'une tête de classification combinée.
    """
    def __init__(self, num_tabular_features: int):
        super(DermascopeMultimodal, self).__init__()
        
        # 1. Branche Visuelle (EfficientNet-B4 ImageNet)
        self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        # Remplacement du classifieur par une couche Identité pour extraire les features
        self.backbone.classifier = nn.Identity()
        
        # Compression des features visuelles
        self.vision_compress = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU()
        )
        
        # 2. Branche Tabulaire (MLP)
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU()
        )
        
        # 3. Tête de Fusion
        # Concaténation: 512 (Vision) + 32 (Tabulaire) = 544
        self.fusion_head = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1) # Sortie pour BCEWithLogitsLoss
        )
        
    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du modèle.
        """
        # Features visuelles
        v_features = self.backbone(images)
        v_features = self.vision_compress(v_features)
        
        # Features tabulaires
        t_features = self.tabular_branch(metadata)
        
        # Fusion
        fused = torch.cat([v_features, t_features], dim=1)
        logits = self.fusion_head(fused)
        
        return logits


class DermascopeFocalLoss(nn.Module):
    """
    Binary Focal Loss optimisée pour les classes déséquilibrées.
    
    Rationnel: Pénalise moins les prédictions sûres et donne un poids 
    supplémentaire aux erreurs de la classe positive (Maligne).
    """
    def __init__(self, alpha: float = config.FOCAL_ALPHA, gamma: float = config.FOCAL_GAMMA):
        super(DermascopeFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # Réduction 'none' pour appliquer la pondération manuelle
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calcule la perte Focal.
        """
        bce_loss = self.bce_with_logits(logits, targets)
        
        # Calcul de pt (probabilité de la vraie classe)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Poids Alpha: on pèse la classe 1 avec alpha, et la classe 0 avec (1-alpha)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Formule de la Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


def create_model(num_tabular_features: int, device: torch.device) -> nn.Module:
    """
    Instancie le modèle multimodal et le charge sur le device.
    
    Args:
        num_tabular_features (int): Nombre de features du vecteur clinique.
        device (torch.device): CPU ou GPU.
        
    Returns:
        nn.Module: Modèle DermaScope.
    """
    model = DermascopeMultimodal(num_tabular_features)
    return model.to(device)
