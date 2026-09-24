# ============================================================
# DermaScope AI — Configuration Centrale du Projet
# ============================================================
# Auteur  : Bilel Kahma
# Projet  : Détection de Pathologies Cutanées par Deep Learning
# Dataset : ISIC HAM10000 (10 015 images dermoscopiques)
# ============================================================

"""
Configuration centralisée pour le projet DermaScope AI.
Tous les hyperparamètres, chemins et constantes sont définis ici.
"""

import os
from pathlib import Path

# ============================================================
# CHEMINS DU PROJET
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Données
TRAIN_CSV = DATA_DIR / "train_df_clean.csv"
VAL_CSV = DATA_DIR / "val_df_clean.csv"
IMAGES_DIR = DATA_DIR / "Images_Dermascope_Propres"

# ============================================================
# HYPERPARAMÈTRES DU MODÈLE
# ============================================================
IMAGE_SIZE = 512          # Résolution standardisée (DullRazor + SAM)
BATCH_SIZE = 8            # Optimisé pour GPU T4 (16 Go VRAM)
ACCUMULATION_STEPS = 2    # Batch effectif = 8 × 2 = 16
NUM_WORKERS = 2
PIN_MEMORY = True

# ============================================================
# ENTRAÎNEMENT
# ============================================================
EPOCHS = 30
PATIENCE = 6              # Early Stopping après 6 époques sans progrès

# Learning Rates Différentiels (Stratégie clé du Transfer Learning)
LR_BACKBONE = 1e-5        # EfficientNet pré-entraîné → micro-ajustements
LR_COMPRESS = 5e-4        # Couche de compression → vitesse moyenne
LR_TABULAR = 1e-3         # Branche tabulaire (neuve) → apprentissage rapide
LR_FUSION = 5e-4          # Tête de fusion → vitesse moyenne
WEIGHT_DECAY = 1e-4

# Scheduler
SCHEDULER_T0 = 5          # CosineAnnealingWarmRestarts : période initiale
SCHEDULER_TMULT = 2       # Multiplicateur de période

# ============================================================
# FOCAL LOSS (Anti-Déséquilibre)
# ============================================================
FOCAL_ALPHA = 0.75        # Poids de la classe minoritaire (Malin)
FOCAL_GAMMA = 2.0         # Facteur de focalisation

# ============================================================
# CLASSIFICATION MÉDICALE
# ============================================================
# Classes malignes (cancéreuses) dans le dataset ISIC HAM10000
MALIGNANT_CLASSES = ["mel", "bcc", "akiec"]

# Mapping des diagnostics
TARGET_LABELS = {
    0: "Bénin (Grain de beauté, tache de vieillesse...)",
    1: "Malin / Suspect (Mélanome, Carcinome, Kératose actinique)",
}

# ============================================================
# NORMALISATION ImageNet (Standard pour Transfer Learning)
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# DATA AUGMENTATION
# ============================================================
# Augmentation standard pour la classe majoritaire (Bénin)
AUG_ROTATION_BENIN = 30
AUG_BRIGHTNESS_BENIN = 0.3
AUG_CONTRAST_BENIN = 0.3

# Augmentation agressive pour la classe minoritaire (Malin)
AUG_ROTATION_MALIN = 360
AUG_TRANSLATE_MALIN = (0.1, 0.1)
AUG_SCALE_MALIN = (0.85, 1.15)
AUG_BRIGHTNESS_MALIN = 0.4
AUG_CONTRAST_MALIN = 0.4

# ============================================================
# DULLRAZOR (Suppression des poils)
# ============================================================
DULLRAZOR_KERNEL_SIZE = 17
DULLRAZOR_BLACKHAT_THRESHOLD = 40
DULLRAZOR_TOPHAT_THRESHOLD = 50
DULLRAZOR_DILATION_SIZE = 5
DULLRAZOR_INPAINT_RADIUS = 5

# ============================================================
# SEED (Reproductibilité)
# ============================================================
RANDOM_SEED = 42
