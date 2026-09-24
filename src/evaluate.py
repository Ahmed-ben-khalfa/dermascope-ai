"""
============================================================
DermaScope AI — Évaluation
============================================================
Auteur  : Bilel Kahma
Projet  : Détection de Pathologies Cutanées par Deep Learning
============================================================
Évaluation du modèle, calcul des métriques médicales, matrice 
de confusion et courbe ROC.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, \
                            specificity_score, roc_auc_score, f1_score, classification_report, \
                            confusion_matrix, roc_curve
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

from src import config
from src.dataset import encoder_rapide, create_dataloaders
from src.model import create_model
from src.train import prepare_data

__all__ = ['load_model', 'evaluate', 'compute_medical_metrics', 'plot_confusion_matrix', 'plot_roc_curve', 'generate_full_report']

def load_model(checkpoint_path: Path, num_tabular_features: int, device: torch.device) -> torch.nn.Module:
    """
    Charge les poids pré-entraînés dans le modèle.
    """
    model = create_model(num_tabular_features, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Passe les données de validation dans le modèle pour générer prédictions et probabilités.
    
    Returns:
        Tuple: (y_true, y_pred, y_probs)
    """
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for images, meta, labels in tqdm(loader, desc="🔍 Évaluation globale"):
            images, meta = images.to(device), meta.to(device)
            
            logits = model(images, meta)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            y_true.extend(labels.cpu().numpy().flatten())
            y_probs.extend(probs.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())
            
    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def compute_medical_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> dict:
    """
    Calcule les métriques essentielles pour un diagnostic médical.
    Rationnel: La sensibilité (Recall) est critique pour ne pas manquer de cancers.
    """
    
    # Specificity is Recall of the negative class (0 = Bénin)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Sensitivity (Recall)': recall_score(y_true, y_pred),
        'Specificity': specificity,
        'ROC-AUC': roc_auc_score(y_true, y_probs),
        'F1-Score': f1_score(y_true, y_pred)
    }
    
    print("\n📊 --- RAPPORT DE CLASSIFICATION ---")
    print(classification_report(y_true, y_pred, target_names=["Bénin (0)", "Malin (1)"]))
    
    print("\n⚕️ --- MÉTRIQUES MÉDICALES CLÉS ---")
    for k, v in metrics.items():
        print(f"🔹 {k}: {v:.4f}")
        
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """
    Génère et sauvegarde la matrice de confusion via Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Bénin", "Malin"], yticklabels=["Bénin", "Malin"])
    plt.title("Matrice de Confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérité Terrain")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, save_path: Path) -> None:
    """
    Trace et sauvegarde la courbe ROC avec calcul de l'AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (1 - Specificité)')
    plt.ylabel('Taux de Vrais Positifs (Sensibilité)')
    plt.title('Courbe ROC - DermaScope AI')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_full_report(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, output_dir: Path) -> None:
    """
    Génère les métriques et les graphiques de performance complets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_true, y_pred, y_probs = evaluate(model, loader, device)
    metrics = compute_medical_metrics(y_true, y_pred, y_probs)
    
    plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, y_probs, output_dir / "roc_curve.png")
    
    print(f"\n📂 Graphiques sauvegardés dans : {output_dir}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Récupération des DataLoaders
    _, _, _, val_meta, num_features = prepare_data()
    
    # Nous n'avons besoin que du val_loader pour l'évaluation (création basique)
    import pandas as pd
    val_df = pd.read_csv(config.VAL_CSV)
    # on bypass la duplication du df, on va juste utiliser create_dataloaders
    train_df = val_df # hack rapide car on evalue juste
    train_loader, val_loader = create_dataloaders(val_df, val_df, val_meta, val_meta, config.BATCH_SIZE)
    
    checkpoint = config.MODELS_DIR / 'best_model.pth'
    if checkpoint.exists():
        model = load_model(checkpoint, num_features, device)
        generate_full_report(model, val_loader, device, config.RESULTS_DIR)
    else:
        print(f"❌ Checkpoint introuvable à : {checkpoint}")
