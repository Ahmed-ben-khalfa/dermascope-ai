"""
============================================================
DermaScope AI — Entraînement
============================================================
Auteur  : Bilel Kahma
Projet  : Détection de Pathologies Cutanées par Deep Learning
============================================================
Pipeline complète d'entraînement avec mixed precision, learning rates
différentiels, Focal Loss, et checkpoints.
"""

import os
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm
from pathlib import Path
from typing import Tuple

from src import config
from src.dataset import encoder_rapide, create_dataloaders
from src.model import create_model, DermascopeFocalLoss

__all__ = ['prepare_data', 'train_one_epoch', 'validate', 'train']

def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Prépare et encode les données d'entraînement et de validation.
    
    Returns:
        Tuple: (train_df, val_df, nombre de features tabulaires).
    """
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV)
    
    # Remplacer les valeurs manquantes (nettoyage basique si pas déjà fait)
    age_median = train_df['age'].median()
    train_df['age'] = train_df['age'].fillna(age_median)
    val_df['age'] = val_df['age'].fillna(age_median)
    
    train_df['sex'] = train_df['sex'].fillna('unknown')
    val_df['sex'] = val_df['sex'].fillna('unknown')
    train_df['localization'] = train_df['localization'].fillna('unknown')
    val_df['localization'] = val_df['localization'].fillna('unknown')
    
    # Normalisation de l'âge
    train_df['age_norm'] = train_df['age'] / 100.0
    val_df['age_norm'] = val_df['age'] / 100.0
    
    # Détermination des catégories
    sex_cats = list(set(train_df['sex'].unique()).union(val_df['sex'].unique()))
    loc_cats = list(set(train_df['localization'].unique()).union(val_df['localization'].unique()))
    
    # Encodage
    train_meta = encoder_rapide(train_df, sex_cats, loc_cats)
    val_meta = encoder_rapide(val_df, sex_cats, loc_cats)
    
    num_features = train_meta.shape[1]
    
    # Attach encodings to dataframes for dataloader simplicity
    train_df['meta_arr'] = list(train_meta)
    val_df['meta_arr'] = list(val_meta)
    
    return train_df, val_df, train_meta, val_meta, num_features


def train_one_epoch(model: torch.nn.Module, loader: torch.utils.data.DataLoader, 
                    criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    scaler: torch.amp.GradScaler, device: torch.device, 
                    accumulation_steps: int) -> float:
    """
    Entraîne le modèle sur une époque entière.
    """
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc="🔄 Entraînement")
    for i, (images, meta, labels) in pbar:
        images, meta, labels = images.to(device), meta.to(device), labels.to(device)
        
        # Mixed Precision
        with torch.amp.autocast(device_type=device.type if device.type == 'cuda' else 'cpu'):
            logits = model(images, meta)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps
            
        # Backward
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
        
    return running_loss / len(loader)


def validate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, 
             criterion: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Évalue le modèle sur le jeu de validation.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, meta, labels in tqdm(loader, desc="🔬 Validation"):
            images, meta, labels = images.to(device), meta.to(device), labels.to(device)
            
            with torch.amp.autocast(device_type=device.type if device.type == 'cuda' else 'cpu'):
                logits = model(images, meta)
                loss = criterion(logits, labels)
                
            running_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    val_loss = running_loss / len(loader)
    val_acc = correct / total
    return val_loss, val_acc


def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, 
          val_loader: torch.utils.data.DataLoader, device: torch.device, 
          epochs: int, patience: int, accumulation_steps: int, checkpoint_dir: Path) -> None:
    """
    Boucle complète d'entraînement avec early stopping et scheduling.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Differential Learning Rates
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': config.LR_BACKBONE},
        {'params': model.vision_compress.parameters(), 'lr': config.LR_COMPRESS},
        {'params': model.tabular_branch.parameters(), 'lr': config.LR_TABULAR},
        {'params': model.fusion_head.parameters(), 'lr': config.LR_FUSION}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.SCHEDULER_T0, T_mult=config.SCHEDULER_TMULT
    )
    
    criterion = DermascopeFocalLoss()
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else torch.amp.GradScaler('cpu')
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("🚀 Début de l'entraînement DermaScope AI")
    
    for epoch in range(1, epochs + 1):
        print(f"\n📅 Époque {epoch}/{epochs}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"📉 Train Loss: {train_loss:.4f} | 📈 Val Loss: {val_loss:.4f} | 🎯 Val Acc: {val_acc*100:.2f}%")
        
        # Checkpoint Last
        torch.save(model.state_dict(), checkpoint_dir / 'last_model.pth')
        
        # Early Stopping & Best Checkpoint
        if val_loss < best_val_loss:
            print("🌟 Nouveau meilleur modèle trouvé ! Sauvegarde...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⚠️ Pas d'amélioration. Patience: {epochs_no_improve}/{patience}")
            
        if epochs_no_improve >= patience:
            print("🛑 Early Stopping déclenché. Arrêt de l'entraînement.")
            break


if __name__ == '__main__':
    # Initialisation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_df, val_df, train_meta, val_meta, num_features = prepare_data()
    train_loader, val_loader = create_dataloaders(train_df, val_df, train_meta, val_meta, config.BATCH_SIZE)
    
    model = create_model(num_features, device)
    
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config.EPOCHS,
        patience=config.PATIENCE,
        accumulation_steps=config.ACCUMULATION_STEPS,
        checkpoint_dir=config.MODELS_DIR
    )
