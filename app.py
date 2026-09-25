"""
Dermascope AI — Interactive Streamlit Dashboard
Real-time melanoma detection with FiLM ensemble + Filtered Grad-CAM.
"""

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import os

from models import Dermascope_FiLM_EfficientNet, Dermascope_FiLM_Alternative
from utils import (
    encode_metadata, predict_tta, GradCAM_Sniper,
    SEX_CATS, LOC_CATS, NUM_TABULAR_FEATURES,
    VAL_TRANSFORM_512, OPTIMAL_THRESHOLD
)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Dermascope AI | Melanoma Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-title {
    font-size: 48px; font-weight: 700; color: #0F172A;
    background: linear-gradient(135deg, #0EA5E9, #6366F1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.subtitle { font-size: 18px; color: #64748B; margin-bottom: 2rem; }
.alert-malignant {
    background: linear-gradient(135deg, #FEE2E2, #FECACA);
    color: #991B1B; padding: 24px; border-radius: 16px;
    text-align: center; font-size: 26px; font-weight: 700;
    border-left: 6px solid #EF4444; margin: 1rem 0;
}
.alert-benign {
    background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
    color: #065F46; padding: 24px; border-radius: 16px;
    text-align: center; font-size: 26px; font-weight: 700;
    border-left: 6px solid #10B981; margin: 1rem 0;
}
.metric-card {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 16px; text-align: center;
}
.metric-value { font-size: 28px; font-weight: 700; color: #0F172A; }
.metric-label { font-size: 12px; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)

# ── Model Loading ────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')


@st.cache_resource
def load_jury():
    """Load all 3 FiLM models from disk."""
    models = {}
    specs = [
        ('effnet', Dermascope_FiLM_EfficientNet, 'best_film_512_effnet.pth'),
        ('resnet', Dermascope_FiLM_Alternative, 'best_film_512_resnet.pth'),
        ('densenet', Dermascope_FiLM_Alternative, 'best_film_512_densenet.pth'),
    ]
    for key, cls, fname in specs:
        path = os.path.join(WEIGHTS_DIR, fname)
        if not os.path.exists(path):
            st.error(f"Weight file not found: {path}")
            return None, None, None
        if key == 'effnet':
            m = cls(NUM_TABULAR_FEATURES).to(DEVICE)
        elif key == 'resnet':
            m = cls(NUM_TABULAR_FEATURES, "resnet50").to(DEVICE)
        else:
            m = cls(NUM_TABULAR_FEATURES, "densenet121").to(DEVICE)
        checkpoint = torch.load(path, map_location=DEVICE)
        m.load_state_dict(checkpoint['model_state'])
        m.eval()
        models[key] = m
    return models['effnet'], models['resnet'], models['densenet']


m1, m2, m3 = load_jury()

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-title">Dermascope AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multimodal Melanoma Detection System — FiLM Architecture + Filtered Grad-CAM</div>', unsafe_allow_html=True)

# ── Performance badges ───────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-label">ROC-AUC</div><div class="metric-value">0.9095</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-label">Sensitivity</div><div class="metric-value">87.0%</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-label">Specificity</div><div class="metric-value">79.2%</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-label">Threshold</div><div class="metric-value">0.461</div></div>', unsafe_allow_html=True)

st.write("---")

# ── Sidebar: Patient Data ────────────────────────────────────
col_input, col_result = st.columns([1, 3])

with col_input:
    st.header("Patient Record")
    age = st.slider("Age", 0, 100, 50)
    sex = st.selectbox("Sex", ["Female", "Male", "unknown"])
    loc = st.selectbox("Lesion Localization", LOC_CATS)

    st.write("---")
    st.header("Dermoscopic Image")
    uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

# ── Main: Results ────────────────────────────────────────────
with col_result:
    if uploaded is not None and m1 is not None:
        image_pil = Image.open(uploaded).convert("RGB")
        img_np = np.array(image_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_512 = cv2.resize(img_bgr, (512, 512))
        img_rgb_512 = cv2.cvtColor(img_512, cv2.COLOR_BGR2RGB)

        # Prepare tensors
        img_t = VAL_TRANSFORM_512(img_512).unsqueeze(0).to(DEVICE)
        meta_vec = encode_metadata(age, sex, loc)
        meta_t = torch.tensor(meta_vec, dtype=torch.float32).to(DEVICE)

        # TTA Ensemble prediction
        with torch.no_grad():
            p1 = predict_tta(m1, img_t, meta_t).item()
            p2 = predict_tta(m2, img_t, meta_t).item()
            p3 = predict_tta(m3, img_t, meta_t).item()
        prob = 0.33 * p1 + 0.33 * p2 + 0.34 * p3

        # Grad-CAM Sniper (on EfficientNet)
        grad_cam = GradCAM_Sniper(m1, m1.vision.features[-1])
        heatmap, _ = grad_cam.generate(img_t, meta_t)

        # Build overlay: original where heatmap=0, JET blend where heatmap>0
        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        mask_3ch = np.stack([heatmap > 0] * 3, axis=-1).astype(np.float32)
        overlay = (
            img_rgb_512.astype(np.float32) * (1 - mask_3ch * 0.5)
            + heatmap_color.astype(np.float32) * mask_3ch * 0.5
        )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Decision
        st.markdown("### Diagnosis Result")
        if prob >= OPTIMAL_THRESHOLD:
            st.markdown(
                f'<div class="alert-malignant">MALIGNANCY ALERT — {prob*100:.1f}% probability</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="alert-benign">LIKELY BENIGN — {prob*100:.1f}% probability</div>',
                unsafe_allow_html=True
            )

        # Display images
        st.write("")
        img_col, cam_col = st.columns(2)
        with img_col:
            st.image(img_rgb_512, caption="Original Dermoscopic Image", use_container_width=True)
        with cam_col:
            st.image(overlay, caption="Grad-CAM Attention Map (Explainability)", use_container_width=True)

        # Individual model scores
        st.write("---")
        st.markdown("#### Individual Model Scores")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("EfficientNet-B4 + FiLM", f"{p1*100:.1f}%")
        with mc2:
            st.metric("ResNet-50 + FiLM", f"{p2*100:.1f}%")
        with mc3:
            st.metric("DenseNet-121 + FiLM", f"{p3*100:.1f}%")

    elif uploaded is None:
        st.info("Upload a dermoscopic image and fill in the patient record to run the diagnosis.")
    else:
        st.error("Models could not be loaded. Place your .pth files in the weights/ folder.")
