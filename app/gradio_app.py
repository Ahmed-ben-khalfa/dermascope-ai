# ============================================================
# DermaScope AI — Application de Démonstration (Gradio)
# ============================================================
# Auteur  : Bilel Kahma
# Projet  : Détection de Pathologies Cutanées par Deep Learning
# ============================================================

"""
Application web de démonstration pour DermaScope AI.
Permet de télécharger une image dermoscopique et d'obtenir
une prédiction Bénin / Malin avec un score de confiance.

Usage :
    python app/gradio_app.py
"""

import sys
from pathlib import Path

# Ajout du répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import gradio as gr
except ImportError:
    print("❌ Gradio non installé. Exécutez : pip install gradio>=4.0.0")
    sys.exit(1)

from src.config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    MODELS_DIR, TARGET_LABELS,
)
from src.model import DermascopeMultimodal


# ============================================================
# CONFIGURATION DE L'APPLICATION
# ============================================================
CHECKPOINT_PATH = MODELS_DIR / "best_multimodal_champion.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_demo_model() -> tuple:
    """
    Charge le modèle entraîné pour la démonstration.

    Returns:
        tuple: (modèle, num_tabular_features) ou (None, 0) si pas de checkpoint.
    """
    if not CHECKPOINT_PATH.exists():
        print(f"⚠️ Aucun modèle trouvé à {CHECKPOINT_PATH}")
        print("   Lancez d'abord l'entraînement avec : python src/train.py")
        return None, 0

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    num_features = checkpoint.get("num_tabular_features", 19)

    model = DermascopeMultimodal(num_tabular_features=num_features).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"✅ Modèle chargé depuis {CHECKPOINT_PATH}")
    return model, num_features


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Prétraite une image pour le modèle.

    Args:
        image: Image RGB en numpy array.

    Returns:
        Tensor normalisé (1, 3, 512, 512).
    """
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32) / 255.0

    # Normalisation ImageNet
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    image = (image - mean) / std

    # HWC → CHW → Batch
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


def predict(image: np.ndarray) -> dict:
    """
    Effectue une prédiction sur une image dermoscopique.

    Args:
        image: Image RGB téléchargée par l'utilisateur.

    Returns:
        Dictionnaire {label: probabilité} pour Gradio.
    """
    if model is None:
        return {"❌ Modèle non chargé": 1.0}

    # Prétraitement de l'image
    img_tensor = preprocess_image(image).to(DEVICE)

    # Vecteur de métadonnées par défaut (inconnu)
    meta_tensor = torch.zeros(1, num_features).to(DEVICE)

    # Prédiction
    with torch.no_grad():
        logits = model(img_tensor, meta_tensor)
        prob = torch.sigmoid(logits).item()

    return {
        "🔴 MALIN / Suspect (Consulter un dermatologue)": prob,
        "✅ Bénin (Grain de beauté normal)": 1.0 - prob,
    }


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================
model, num_features = load_demo_model()

# ============================================================
# INTERFACE GRADIO
# ============================================================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="📸 Image dermoscopique"),
    outputs=gr.Label(num_top_classes=2, label="🔬 Diagnostic AI"),
    title="🔬 DermaScope AI — Détection de Pathologies Cutanées",
    description=(
        "**Prototype éducatif** — Système de pré-screening par Deep Learning "
        "multimodal (EfficientNet-B4 + métadonnées cliniques).\n\n"
        "⚠️ **Ce système ne remplace PAS un avis médical.** "
        "Consultez toujours un dermatologue pour toute lésion suspecte."
    ),
    article=(
        "### 📖 À propos\n"
        "- **Dataset** : ISIC HAM10000 (10 015 images)\n"
        "- **Pipeline** : DullRazor → SAM → Bokeh → EfficientNet-B4\n"
        "- **Architecture** : Multimodal (Vision + Tabulaire)\n"
        "- **Auteur** : Bilel Kahma\n"
    ),
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(share=False)
