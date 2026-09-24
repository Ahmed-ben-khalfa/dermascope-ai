# 🔬 DermaScope AI — Détection de Pathologies Cutanées par Deep Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-ISIC%20HAM10000-orange)](https://www.isic-archive.com/)

> **⚠️ AVERTISSEMENT MÉDICAL** : Ce projet est un exercice d'apprentissage et un prototype de recherche. Il n'est **PAS** un dispositif médical certifié (CE/FDA). Il ne remplace **JAMAIS** un dermatologue. Toute lésion suspecte nécessite une consultation médicale et potentiellement une biopsie.

---

## 📋 Description

**DermaScope AI** est un système d'Intelligence Artificielle multimodal de détection de pathologies cutanées, développé dans le cadre d'un cours de Deep Learning. Le système combine :

- **Vision par Ordinateur (Computer Vision)** : Analyse d'images dermoscopiques via EfficientNet-B4
- **Données Cliniques Structurées** : Métadonnées patient (âge, sexe, localisation anatomique)
- **Architecture de Fusion** : Réseau en "Y" combinant les deux modalités pour un diagnostic robuste

Le modèle classe les lésions cutanées en **Bénin (0)** ou **Malin/Suspect (1)** (Mélanome, Carcinome, Kératose actinique).

---

## 🏗️ Architecture

```
                    📸 Image (512×512×3)         📋 Métadonnées (âge, sexe, localisation)
                           │                                    │
                    ┌──────▼──────┐                    ┌───────▼────────┐
                    │ EfficientNet │                    │   MLP Branch   │
                    │    B4        │                    │  (Dense×2 +    │
                    │ (ImageNet)   │                    │   BatchNorm)   │
                    └──────┬──────┘                    └───────┬────────┘
                           │ 1792                              │ 32
                    ┌──────▼──────┐                             │
                    │  Compress   │                             │
                    │ (→ 512)     │                             │
                    └──────┬──────┘                             │
                           │ 512                               │
                           └─────────────┬─────────────────────┘
                                         │ 544
                                  ┌──────▼──────┐
                                  │  Fusion Head │
                                  │ (Dense + DO) │
                                  └──────┬──────┘
                                         │
                                    ┌────▼────┐
                                    │ Sigmoid  │
                                    │ 0=Bénin  │
                                    │ 1=Malin  │
                                    └─────────┘
```

---

## 🔧 Pipeline de Traitement d'Image

Avant d'atteindre le modèle, chaque image passe par un pipeline de 3 étapes conçu pour **détruire les biais visuels** :

### 1. 🪒 DullRazor Universel (Suppression des poils)
- Détection des poils clairs **et** sombres via combinaison Black-Hat + Top-Hat
- Dilatation pour capturer les intersections de poils
- Inpainting (cicatrisation numérique) via l'algorithme de Telea

### 2. 🎯 Segmentation SAM avec Viseur Hybride
- **Anomalie colorimétrique** : Distance LAB par rapport à la peau saine (indépendante du phototype)
- **Gravité centrale** : Biais gaussien pour les images zoomées
- **Fusion** : Anomalie × Gravité → Point Prompt pour SAM (Meta)

### 3. 📷 Effet Bokeh Médical (Anti-Biais de Contour)
- Flou d'arrière-plan pour forcer l'IA à regarder la lésion
- **Feathering** : Le masque est lui-même flouté pour supprimer la ligne de fracture SAM
- **Alpha Blending** : Transition douce et biologique

---

## 📊 Dataset — ISIC HAM10000

| Diagnostic | Type | Danger | Images (~) |
|---|---|---|---|
| `nv` — Naevus (Grain de beauté) | Bénin ✅ | Faible | ~6 705 |
| `mel` — **Mélanome** | **Malin ⚠️** | **MORTEL** | ~1 113 |
| `bkl` — Kératose bénigne | Bénin ✅ | Faible | ~1 099 |
| `bcc` — Carcinome basocellulaire | **Malin ⚠️** | Sérieux | ~514 |
| `akiec` — Kératose actinique | **Pré-malin ⚠️** | Modéré | ~327 |
| `vasc` — Lésion vasculaire | Bénin ✅ | Faible | ~142 |
| `df` — Dermatofibrome | Bénin ✅ | Faible | ~115 |

> **Défi majeur** : Déséquilibre extrême (67% de grains de beauté). Résolu par Focal Loss + Augmentation ciblée.

---

## 🛡️ Stratégies Anti-Déséquilibre

| Stratégie | Implémentation |
|---|---|
| **Focal Loss** | α=0.75 pour la classe Malin, γ=2.0 — pénalise fortement les erreurs sur les cas rares |
| **Augmentation ciblée** | Rotation 360°, zoom, affine, jitter agressif uniquement sur les cas malins |
| **GroupShuffleSplit** | Partitionnement par `patient_id` pour zéro Data Leakage |
| **Learning Rates différentiels** | Backbone (1e-5) vs Tête (1e-3) — protège les features pré-entraînées |

---

## 🚀 Installation & Utilisation

### Prérequis
```bash
Python >= 3.10
CUDA >= 11.8 (recommandé pour l'entraînement)
```

### Installation
```bash
git clone https://github.com/bilel-kahma/dermascope-ai.git
cd dermascope-ai
pip install -r requirements.txt
```

### Entraînement
```bash
python src/train.py
```

### Évaluation
```bash
python src/evaluate.py
```

---

## 📁 Structure du Projet

```
dermascope-ai/
├── README.md                          # Ce fichier
├── requirements.txt                   # Dépendances Python
├── .gitignore                         # Fichiers exclus de Git
├── LICENSE                            # Licence MIT
│
├── data/                              # Données (exclues de Git)
│   ├── train_df_clean.csv             # Métadonnées d'entraînement
│   ├── val_df_clean.csv               # Métadonnées de validation
│   └── Images_Dermascope_Propres/     # Images nettoyées (DullRazor + SAM + Bokeh)
│
├── src/                               # Code source principal
│   ├── config.py                      # Configuration centralisée
│   ├── preprocessing.py               # Pipeline DullRazor → SAM → Bokeh
│   ├── dataset.py                     # Dataset PyTorch multimodal
│   ├── model.py                       # Architecture DermascopeMultimodal
│   ├── train.py                       # Boucle d'entraînement
│   └── evaluate.py                    # Métriques et visualisations
│
├── notebooks/                         # Notebooks Jupyter / Colab
│   └── dermascope_colab_brut.ipynb    # Notebook Colab original
│
├── models/                            # Poids entraînés (.pth)
├── results/                           # Graphiques et rapports générés
├── app/                               # Application Gradio (démo)
│   └── gradio_app.py
│
└── docs/                              # Documentation technique
    ├── rapport_phase1_dermascope.md    # Phase 1 : Data Engineering
    ├── rapport_phase2_dermascope.md    # Phase 2 : Pipeline Image
    ├── explications_architecture.md   # Vulgarisation du réseau
    └── roadmap_multimodale.md         # Feuille de route
```

---

## 🎯 Compétences Démontrées

| Domaine | Compétences |
|---|---|
| **Deep Learning** | Transfer Learning (EfficientNet-B4), Fine-Tuning, Focal Loss, Mixed Precision (FP16), Gradient Accumulation |
| **Computer Vision** | Morphologie mathématique (Black-Hat/Top-Hat), Inpainting, Segmentation (SAM), Alpha Blending |
| **Data Engineering** | Ingestion web (bypass 403), Nettoyage, Imputation, One-Hot Encoding vectorisé, GroupShuffleSplit anti-leakage |
| **Architecture ML** | Modèle multimodal (Vision + Tabulaire), Fusion par concaténation, Learning rates différentiels |
| **MLOps** | Checkpointing (Google Drive), Early Stopping, Reproductibilité (seeds), CosineAnnealing |
| **Éthique IA Médicale** | Biais de phototype, invariance d'échelle, anti-Edge Bias, avertissement médical |

---

## 📚 Technologies

- **Framework** : PyTorch 2.x
- **Backbone** : EfficientNet-B4 (pré-entraîné ImageNet)
- **Segmentation** : Segment Anything Model (SAM) — Meta AI
- **Traitement d'image** : OpenCV (Morphologie, Inpainting, LAB)
- **Data Science** : Pandas, NumPy, Scikit-learn
- **Visualisation** : Matplotlib, Seaborn
- **Démo** : Gradio

---

## 📖 Considérations Éthiques

- Ce système **N'EST PAS** certifié comme dispositif médical (CE/FDA)
- Il ne remplace **JAMAIS** un dermatologue qualifié
- Le dataset ISIC est biaisé vers les peaux claires (Fitzpatrick I-III)
- Notre pipeline de segmentation utilise l'espace LAB pour être **indépendant du phototype**
- Toute lésion suspecte nécessite une **biopsie** et un avis médical professionnel

---

## 👤 Auteur

**Bilel Kahma** — Étudiant en Ingénierie & Data Science

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
