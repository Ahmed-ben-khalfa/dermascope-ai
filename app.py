"""
Dermascope AI — Clinical Analysis Dashboard
Melanoma detection with FiLM ensemble.
"""

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import os
import time

from models import Dermascope_FiLM_EfficientNet, Dermascope_FiLM_Alternative
from utils import (
    encode_metadata, predict_tta,
    SEX_CATS, LOC_CATS, NUM_TABULAR_FEATURES,
    VAL_TRANSFORM_512, OPTIMAL_THRESHOLD
)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Dermascope AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Clean Medical CSS ────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600;700&display=swap');
    
    /* === GLOBAL — Warm light background === */
    .stApp {
        background-color: #faf9f7;
        font-family: 'DM Sans', sans-serif;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; max-width: 1200px; }
    
    /* === SIDEBAR — Deep forest green === */
    section[data-testid="stSidebar"] {
        background-color: #1B4332;
    }
    section[data-testid="stSidebar"] * {
        color: #d8f3dc !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stFileUploader label {
        color: #95d5b2 !important;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stFileUploader > div > div {
        background: rgba(255,255,255,0.08) !important;
        border: 2px dashed rgba(149,213,178,0.3) !important;
        border-radius: 12px !important;
    }
    
    /* === HERO === */
    .hero {
        text-align: center;
        padding: 3rem 2rem;
        margin-bottom: 0.5rem;
    }
    .hero-eyebrow {
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: #2D6A4F;
        font-weight: 700;
    }
    .hero-name {
        font-family: 'DM Serif Display', serif;
        font-size: 56px;
        color: #1B4332;
        margin: 8px 0;
        line-height: 1.1;
    }
    .hero-quote {
        font-family: 'DM Serif Display', serif;
        font-style: italic;
        font-size: 18px;
        color: #6b7280;
        max-width: 600px;
        margin: 16px auto 0 auto;
        line-height: 1.6;
    }
    .hero-author {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* === DIVIDER === */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #d1d5db, transparent);
        margin: 1.5rem 0;
    }
    
    /* === RESULT CARD — MALIGNANT === */
    .verdict-malignant {
        background: #fff;
        border: 2px solid #dc2626;
        border-radius: 20px;
        padding: 36px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(220,38,38,0.08);
        margin: 1rem 0;
    }
    .verdict-malignant .v-badge {
        display: inline-block;
        background: #fef2f2;
        color: #dc2626;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 6px 16px;
        border-radius: 20px;
        margin-bottom: 12px;
    }
    .verdict-malignant .v-title {
        font-family: 'DM Serif Display', serif;
        font-size: 32px;
        color: #991b1b;
        margin: 8px 0;
    }
    .verdict-malignant .v-prob {
        font-size: 56px;
        font-weight: 800;
        color: #dc2626;
        margin: 4px 0;
    }
    .verdict-malignant .v-note {
        font-size: 13px;
        color: #9ca3af;
        margin-top: 8px;
    }
    
    /* === RESULT CARD — BENIGN === */
    .verdict-benign {
        background: #fff;
        border: 2px solid #2D6A4F;
        border-radius: 20px;
        padding: 36px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(45,106,79,0.08);
        margin: 1rem 0;
    }
    .verdict-benign .v-badge {
        display: inline-block;
        background: #f0fdf4;
        color: #2D6A4F;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 6px 16px;
        border-radius: 20px;
        margin-bottom: 12px;
    }
    .verdict-benign .v-title {
        font-family: 'DM Serif Display', serif;
        font-size: 32px;
        color: #1B4332;
        margin: 8px 0;
    }
    .verdict-benign .v-prob {
        font-size: 56px;
        font-weight: 800;
        color: #2D6A4F;
        margin: 4px 0;
    }
    .verdict-benign .v-note {
        font-size: 13px;
        color: #9ca3af;
        margin-top: 8px;
    }
    
    /* === IMAGE FRAME === */
    .img-frame {
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    .img-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #9ca3af;
        font-weight: 700;
        text-align: center;
        margin-bottom: 8px;
    }
    
    /* === MODEL SCORES === */
    .scores-container {
        display: flex;
        gap: 12px;
        margin-top: 1.5rem;
    }
    .score-card {
        flex: 1;
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 8px rgba(0,0,0,0.03);
    }
    .score-name {
        font-size: 11px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .score-value {
        font-family: 'DM Serif Display', serif;
        font-size: 28px;
        color: #1B4332;
        margin: 6px 0;
    }
    .score-bar {
        height: 4px;
        background: #f3f4f6;
        border-radius: 2px;
        overflow: hidden;
        margin-top: 8px;
    }
    .score-fill {
        height: 100%;
        border-radius: 2px;
        background: #2D6A4F;
    }
    
    /* === WAITING STATE === */
    .waiting {
        text-align: center;
        padding: 80px 20px;
    }
    .waiting-icon {
        font-size: 56px;
        margin-bottom: 16px;
    }
    .waiting h2 {
        font-family: 'DM Serif Display', serif;
        font-size: 28px;
        color: #1B4332 !important;
        margin-bottom: 8px;
    }
    .waiting p {
        font-size: 15px;
        color: #9ca3af;
        max-width: 420px;
        margin: 0 auto;
        line-height: 1.7;
    }
    
    /* === STREAMLIT OVERRIDES === */
    .stMarkdown p { color: #374151; }
    h1, h2, h3 { color: #1B4332 !important; }
    .stImage > img { border-radius: 12px; }
    .stProgress > div > div { background-color: #2D6A4F !important; }
</style>
""", unsafe_allow_html=True)

# ── Model Loading ────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')


@st.cache_resource
def load_jury():
    models = {}
    specs = [
        ('effnet', Dermascope_FiLM_EfficientNet, 'best_film_512_effnet.pth'),
        ('resnet', Dermascope_FiLM_Alternative, 'best_film_512_resnet.pth'),
        ('densenet', Dermascope_FiLM_Alternative, 'best_film_512_densenet.pth'),
    ]
    for key, cls, fname in specs:
        path = os.path.join(WEIGHTS_DIR, fname)
        if not os.path.exists(path):
            return None, None, None
        if key == 'effnet':
            m = cls(NUM_TABULAR_FEATURES).to(DEVICE)
        elif key == 'resnet':
            m = cls(NUM_TABULAR_FEATURES, "resnet50").to(DEVICE)
        else:
            m = cls(NUM_TABULAR_FEATURES, "densenet121").to(DEVICE)
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
        m.load_state_dict(checkpoint['model_state'])
        m.eval()
        models[key] = m
    return models.get('effnet'), models.get('resnet'), models.get('densenet')


m1, m2, m3 = load_jury()

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 1rem 0;">
        <div style="font-size:32px;">🩺</div>
        <div style="font-family:'DM Serif Display',serif; font-size:22px; color:#d8f3dc !important; margin-top:6px;">Dermascope AI</div>
        <div style="font-size:10px; color:#95d5b2 !important; letter-spacing:2px; text-transform:uppercase;">Clinical Analysis</div>
    </div>
    <hr style="border:none; height:1px; background:rgba(255,255,255,0.1); margin:0.5rem 0 1.5rem 0;">
    """, unsafe_allow_html=True)

    age = st.slider("Patient Age", 0, 100, 50)
    sex = st.selectbox("Sex", ["Female", "Male", "unknown"])
    loc = st.selectbox("Lesion Localization", LOC_CATS)

    st.markdown("<hr style='border:none; height:1px; background:rgba(255,255,255,0.1); margin:1.5rem 0;'>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload dermoscopic image", type=["jpg", "png", "jpeg"])

    st.markdown(f"""
    <hr style="border:none; height:1px; background:rgba(255,255,255,0.1); margin:1.5rem 0;">
    <div style="font-size:10px; color:#74c69d !important; line-height:2;">
        Device: {'GPU' if DEVICE.type == 'cuda' else 'CPU'}<br>
        Models: {'3/3 Loaded' if m1 is not None else 'Not loaded'}<br>
        Threshold: {OPTIMAL_THRESHOLD:.4f}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

# Hero Section
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multimodal Deep Learning</div>
    <div class="hero-name">Dermascope AI</div>
    <div class="hero-quote">
        "The earlier you detect melanoma, the better your chance of a cure. 
        Artificial intelligence should serve as an ever-vigilant second pair of eyes."
    </div>
    <div class="hero-author">— Inspired by the American Academy of Dermatology</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Analysis ─────────────────────────────────────────────────
if uploaded is not None and m1 is not None:
    # 1. Save uploaded file temporarily for DullRazor
    temp_path = "temp_uploaded.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())
        
    # 2. Run DullRazor Preprocessing
    from src.preprocessing import remove_hair_dullrazor
    orig_bgr, hair_mask, inpainted_bgr = remove_hair_dullrazor(temp_path)
    
    # Convert back to RGB for display and model
    img_orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    img_clean_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. Prepare for Model
    img_t = VAL_TRANSFORM_512(img_clean_rgb).unsqueeze(0).to(DEVICE)
    meta_vec = encode_metadata(age, sex, loc)
    meta_t = torch.tensor(meta_vec, dtype=torch.float32).to(DEVICE)

    progress = st.progress(0)
    with torch.no_grad():
        progress.progress(15, "Analyzing with EfficientNet-B4...")
        p1 = predict_tta(m1, img_t, meta_t).item()
        progress.progress(45, "Analyzing with ResNet-50...")
        p2 = predict_tta(m2, img_t, meta_t).item()
        progress.progress(75, "Analyzing with DenseNet-121...")
        p3 = predict_tta(m3, img_t, meta_t).item()
        progress.progress(90, "Computing ensemble score...")

    prob = 0.33 * p1 + 0.33 * p2 + 0.34 * p3

    progress.progress(100, "Done.")
    time.sleep(0.3)
    progress.empty()

    # ── VERDICT ──
    if prob >= OPTIMAL_THRESHOLD:
        st.markdown(f"""
        <div class="verdict-malignant">
            <div class="v-badge">⚠ High Risk</div>
            <div class="v-title">Suspected Malignancy</div>
            <div class="v-prob">{prob*100:.1f}%</div>
            <div class="v-note">Probability exceeds the optimal clinical threshold of {OPTIMAL_THRESHOLD:.0%} — Dermatologist review and biopsy recommended</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-benign">
            <div class="v-badge">✓ Low Risk</div>
            <div class="v-title">Likely Benign Lesion</div>
            <div class="v-prob">{prob*100:.1f}%</div>
            <div class="v-note">Below the clinical threshold of {OPTIMAL_THRESHOLD:.0%} — Routine follow-up monitoring is advised</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Images ──
    # Show Original and Cleaned Images side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="img-frame"><div class="img-title">Original (With Hair/Noise)</div></div>', unsafe_allow_html=True)
        st.image(img_orig_rgb, use_container_width=True)
    with col2:
        st.markdown('<div class="img-frame"><div class="img-title">Cleaned (DullRazor)</div></div>', unsafe_allow_html=True)
        st.image(img_clean_rgb, use_container_width=True)

    # ── Model Scores ──
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="scores-container">
        <div class="score-card">
            <div class="score-name">EfficientNet-B4</div>
            <div class="score-value">{p1*100:.1f}%</div>
            <div class="score-bar"><div class="score-fill" style="width:{min(p1*100, 100):.0f}%"></div></div>
        </div>
        <div class="score-card">
            <div class="score-name">ResNet-50</div>
            <div class="score-value">{p2*100:.1f}%</div>
            <div class="score-bar"><div class="score-fill" style="width:{min(p2*100, 100):.0f}%"></div></div>
        </div>
        <div class="score-card">
            <div class="score-name">DenseNet-121</div>
            <div class="score-value">{p3*100:.1f}%</div>
            <div class="score-bar"><div class="score-fill" style="width:{min(p3*100, 100):.0f}%"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Details ──
    with st.expander("Technical Details"):
        st.markdown(f"""
        | | |
        |---|---|
        | **Fusion** | FiLM — Feature-wise Linear Modulation |
        | **Ensemble** | Weighted Average (0.33 / 0.33 / 0.34) |
        | **TTA** | 5 augmentations per model |
        | **Threshold** | {OPTIMAL_THRESHOLD:.4f} (Youden Index) |
        | **Patient** | Age {age}, {sex}, {loc} |
        """)

elif uploaded is None:
    st.markdown("""
    <div class="waiting">
        <div class="waiting-icon">🩺</div>
        <h2>Ready for Clinical Analysis</h2>
        <p>
            Upload a dermoscopic image in the sidebar and provide the patient's clinical information. 
            The AI jury of three expert models will analyze the lesion in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("Models not loaded. Place your .pth files in the weights/ folder.")
