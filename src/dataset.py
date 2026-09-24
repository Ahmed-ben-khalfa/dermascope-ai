"""
============================================================
DermaScope AI — Chargement des Données
============================================================
Auteur  : Bilel Kahma
Projet  : Détection de Pathologies Cutanées par Deep Learning
============================================================
Modules de chargement, encodage et augmentation des données.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, List, Callable, Optional

from src import config

__all__ = ['encoder_rapide', 'DermascopeMultimodalDataset', 'get_transforms', 'create_dataloaders']

def encoder_rapide(df: pd.DataFrame, sex_cats: List[str], loc_cats: List[str]) -> np.ndarray:
    """
    Encode rapidement les métadonnées cliniques sous forme de vecteurs continus/one-hot.
    
    Rationnel: Les métadonnées sont cruciales pour le diagnostic. L'âge est normalisé,
    le sexe et la localisation sont encodés en One-Hot.
    
    Args:
        df (pd.DataFrame): DataFrame contenant 'age_norm', 'sex', 'localization'.
        sex_cats (List[str]): Liste des catégories de sexes.
        loc_cats (List[str]): Liste des catégories de localisations.
        
    Returns:
        np.ndarray: Matrice NumPy des caractéristiques (N, num_features).
    """
    age_feature = df['age_norm'].values.reshape(-1, 1)
    
    # Encodage One-hot optimisé
    sex_dummies = pd.get_dummies(pd.Categorical(df['sex'], categories=sex_cats))
    loc_dummies = pd.get_dummies(pd.Categorical(df['localization'], categories=loc_cats))
    
    meta_matrix = np.hstack([age_feature, sex_dummies.values, loc_dummies.values]).astype(np.float32)
    return meta_matrix


class DermascopeMultimodalDataset(Dataset):
    """
    Dataset PyTorch pour la fusion multimodale (Images + Données cliniques tabulaires).
    
    Rationnel: Applique une data augmentation agressive sélectivement sur la classe
    maligne pour balancer les échantillons au niveau visuel.
    """
    def __init__(self, df: pd.DataFrame, metadata_matrix: np.ndarray, transform: Optional[Callable] = None, transform_malin: Optional[Callable] = None):
        """
        Initialisation du Dataset.
        
        Args:
            df (pd.DataFrame): DataFrame contenant 'path' et 'target'.
            metadata_matrix (np.ndarray): Matrice des métadonnées (encodées).
            transform (Callable, optional): Transformations standard.
            transform_malin (Callable, optional): Transformations agressives pour classe positive.
        """
        self.paths = df['path'].values
        self.labels = df['target'].values
        self.metadata = metadata_matrix
        self.transform = transform
        self.transform_malin = transform_malin

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Récupère l'image et ses métadonnées.
        """
        img_path = str(self.paths[idx])
        label = self.labels[idx]
        
        # Chargement image
        image = cv2.imread(img_path)
        if image is None:
            # Sécurité si chemin invalide (génère une image noire)
            image = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        
        from PIL import Image
        image_pil = Image.fromarray(image)
        
        # Augmentation sélective
        if label == 1 and self.transform_malin is not None:
            image_tensor = self.transform_malin(image_pil)
        elif self.transform is not None:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = transforms.ToTensor()(image_pil)
            
        meta_tensor = torch.tensor(self.metadata[idx], dtype=torch.float32)
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return image_tensor, meta_tensor, label_tensor


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """
    Renvoie les pipelines de transformations (Standard, Malin, Val).
    
    Rationnel: Les lésions bénignes reçoivent une augmentation légère, tandis que
    les lésions malignes (minoritaires) subissent des transformations importantes.
    
    Returns:
        Tuple: (train_transforms, train_transforms_malin, val_transforms).
    """
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(config.AUG_ROTATION_BENIN),
        transforms.ColorJitter(brightness=config.AUG_BRIGHTNESS_BENIN, contrast=config.AUG_CONTRAST_BENIN),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    train_transforms_malin = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(config.AUG_ROTATION_MALIN),
        transforms.RandomAffine(degrees=0, translate=config.AUG_TRANSLATE_MALIN, scale=config.AUG_SCALE_MALIN),
        transforms.ColorJitter(brightness=config.AUG_BRIGHTNESS_MALIN, contrast=config.AUG_CONTRAST_MALIN),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    
    return train_transforms, train_transforms_malin, val_transforms


def create_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, train_meta: np.ndarray, val_meta: np.ndarray, batch_size: int = config.BATCH_SIZE) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les DataLoaders PyTorch pour l'entraînement et la validation.
    
    Args:
        train_df (pd.DataFrame): DataFrame d'entraînement.
        val_df (pd.DataFrame): DataFrame de validation.
        train_meta (np.ndarray): Matrice des métadonnées d'entraînement.
        val_meta (np.ndarray): Matrice des métadonnées de validation.
        batch_size (int): Taille de lot.
        
    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader).
    """
    trans_std, trans_malin, trans_val = get_transforms()
    
    train_dataset = DermascopeMultimodalDataset(train_df, train_meta, transform=trans_std, transform_malin=trans_malin)
    val_dataset = DermascopeMultimodalDataset(val_df, val_meta, transform=trans_val, transform_malin=None)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader
