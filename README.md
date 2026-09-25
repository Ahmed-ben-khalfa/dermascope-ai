<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&pause=1000&color=0E75B6&center=true&vCenter=true&width=700&lines=Dermascope+AI+%F0%9F%94%AC;Multimodal+Deep+Learning;Early+Melanoma+Detection;ROC-AUC%3A+0.9095+%7C+Sensitivity%3A+87.0%25" alt="Typing SVG" />

<br/>

<p>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/ROC--AUC-0.9095-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/Sensitivity-87.02%25-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/Specificity-79.15%25-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/Architecture-FiLM%20Ensemble-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Training-Progressive%20Resizing-orange?style=flat-square"/>
</p>

</div>

---

## What is Dermascope AI?

**Dermascope AI** is a multimodal deep learning system for early melanoma detection. It fuses **high-resolution dermoscopic imagery** with **structured patient clinical data** (age, sex, lesion localization) at the feature extraction level using **Feature-wise Linear Modulation (FiLM)**.

The model doesn't just look at a skin lesion — it looks at the lesion **through the lens of who the patient is**.

> **Clinical Goal:** Provide dermatologists with a high-sensitivity AI second opinion to reduce missed diagnoses during skin cancer screening.

---

## Performance

| Metric | Score |
|--------|:---:|
| **ROC-AUC** | **0.9095** |
| **Sensitivity (Recall)** | **87.02%** |
| **Specificity** | **79.15%** |
| **Optimal Threshold (Youden)** | **0.4607** |

### Classification Report (Optimal Threshold)

```
              precision    recall  f1-score   support

      Benign     0.9620    0.7915    0.8685      1631
   Malignant     0.5015    0.8702    0.6363       393

    accuracy                         0.8068      2024
```

---

## Core Innovation: FiLM Conditioning

Standard multimodal approaches concatenate image features and metadata at the classifier. This means the CNN processes images **blindly**, with no knowledge of the patient.

FiLM solves this by injecting metadata directly into the visual feature extraction:

```
Metadata ──► [MLP] ──► [32-d conditioning vector]
                                    │
                               γ and β
                                    │
Image ──► [CNN] ──► [Pool] ──► [Compress 512-d] ──► [FiLM ⊗ +] ──► [Classify] ──► Prediction

FiLM: output = vision × (1 + γ(metadata)) + β(metadata)
```

The network learns to **re-weight its visual attention** based on the patient's clinical profile.

---

## Architecture: Three-Expert Ensemble

| Model | Individual AUC | Ensemble Weight |
|-------|:---:|:---:|
| EfficientNet-B4 + FiLM | 0.893 | 0.33 |
| ResNet-50 + FiLM | 0.896 | 0.33 |
| DenseNet-121 + FiLM | 0.909 | 0.34 |
| **Weighted TTA Ensemble** | **0.9095** | — |

Each model follows the same pipeline:
1. **Pretrained CNN Backbone** (ImageNet) → visual feature extraction
2. **Compression Layer** (→ 512-d) → standardized feature space
3. **Tabular MLP** (→ 64 → 32-d) → clinical metadata encoding
4. **FiLM Layer** → metadata-conditioned affine modulation
5. **Classifier Head** (512 → 256 → 1) → binary prediction

---

## Training Strategy: Progressive Resizing

```
Phase 1: Warm-Up              Phase 2: Fine-Tuning
─────────────────────          ─────────────────────
Resolution: 256×256            Resolution: 512×512
Backbone:   FROZEN             Backbone:   UNFROZEN
LR:         1e-3               LR:         1e-4
Batch:      32                 Batch:      8
Epochs:     15 (patience 6)    Epochs:     10 (patience 4)
─────────────────────          ─────────────────────
Goal: Train new layers         Goal: Fine-tune backbone
without destroying pretrained  at full clinical resolution
ImageNet representations.      for cellular micro-detail.
```

**Loss Function:** Focal Loss (α=0.75, γ=2.0) — down-weights easy benign samples, concentrates on hard malignant cases.

---



1. **Normalization** → scale activations to [0, 1]
2. **Center-weighted Gaussian mask** → suppress corner/edge artifacts from zero-padding
3. **Aggressive threshold (0.45)** → retain only the strongest activations
4. **Morphological cleanup** → remove isolated noise, fill holes

The result: heatmaps that precisely outline the lesion, ignoring healthy skin entirely.

---

## Quick Start

### 1. Clone
```bash
git clone https://github.com/Ahmed-ben-khalfa/dermascope-ai.git
cd dermascope-ai
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Set Up Weights
Place your trained `.pth` files in the `weights/` directory:
```
weights/
├── best_film_512_effnet.pth
├── best_film_512_resnet.pth
└── best_film_512_densenet.pth
```

### 4. Launch
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Repository Structure

```
dermascope-ai/
├── app.py                         # Streamlit interactive dashboard
├── models.py                      # FiLM architecture (3 backbones)
├── requirements.txt               # Python dependencies
├── README.md
├── REPORT_Dermascope_AI_Full.md   # Full technical research report
│
├── src/                           # Training pipeline (Kaggle)
│   ├── config.py                  # Centralized hyperparameters
│   ├── dataset.py                 # PyTorch Dataset + DataLoader
│   ├── model.py                   # Model definitions
│   ├── train.py                   # Training loop with early stopping
│   ├── evaluate.py                # Evaluation + ROC + confusion matrix
│   └── preprocessing.py           # DullRazor hair removal
│
├── data/                          # CSV annotations (images excluded)
│   ├── train_df_clean.csv
│   └── val_df_clean.csv
│
├── weights/                       # Trained model checkpoints
│   ├── best_film_512_effnet.pth
│   ├── best_film_512_resnet.pth
│   └── best_film_512_densenet.pth
│
├── docs/                          # Architecture diagrams + reports
└── notebooks/                     # Kaggle training notebook
```

---

## Training Environment

| Component | Specification |
|-----------|------|
| GPU | NVIDIA Tesla T4 (16GB VRAM) |
| Framework | PyTorch 2.x + Mixed Precision (AMP) |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=5) |
| Gradient Clipping | max_norm=1.0 |
| Total Training Time | ~3.5 hours (both phases, all 3 models) |

---

## Key Design Decisions

<details>
<summary><b>Why FiLM over Cross-Attention?</b></summary>
Cross-attention requires significantly more compute and data. FiLM adds only 2 linear layers (gamma, beta) — minimal overhead, proven convergence on moderate datasets, ideal for Kaggle T4 constraints.
</details>

<details>
<summary><b>Why threshold 0.46 instead of 0.50?</b></summary>
In medical screening, the cost of a False Negative (missed cancer) vastly exceeds the cost of a False Positive (unnecessary biopsy). The Youden Index optimization on the ROC curve found 0.4607 as the point maximizing Sensitivity + Specificity jointly.
</details>

<details>
<summary><b>Why SiLU activation throughout?</b></summary>
EfficientNet internally uses SiLU/Swish. Maintaining the same activation in our compression and classifier layers preserves gradient flow consistency across the architecture boundary.
</details>

<details>
<summary><b>Why Cosine Annealing with Warm Restarts?</b></summary>
The warm restart mechanism helps escape sharp local minima during Phase 1. During Phase 2, the cosine decay smoothly reduces LR, protecting fine-grained 512×512 features from overshooting.
</details>

---

## Author

**Ahmed Ben Khalfa**
AI & Deep Learning Engineer

[GitHub](https://github.com/Ahmed-ben-khalfa) · [LinkedIn](https://www.linkedin.com/in/ahmedbenkhalfa)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<i>"The goal is not to replace the dermatologist. The goal is to make sure no melanoma goes unnoticed."</i>
</div>
