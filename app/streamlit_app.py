"""
AI-Trust Forensic Suite — Streamlit Dashboard
================================================
Drag-and-drop forensic analysis tool for detecting AI-generated images.
Provides authenticity confidence, generator attribution, and interactive heatmaps.

Run: streamlit run app/streamlit_app.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.model import AIForensicModel
from src.utils import (
    get_eval_transforms,
    get_device,
    LABEL_A_NAMES,
    LABEL_B_NAMES,
    IMAGE_SIZE,
)
from src.explainability import (
    compute_attention_rollout,
    attention_rollout_to_heatmap,
    compute_saliency_map,
    create_heatmap_overlay,
)


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="AI-Trust Forensic Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ──────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #FFFFFF;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1rem;
        margin: 0;
    }

    /* Verdict Card */
    .verdict-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .verdict-authentic {
        border-left: 4px solid #00E676;
    }
    .verdict-synthetic {
        border-left: 4px solid #FF5252;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(255, 255, 255, 0.08);
    }

    /* Sidebar */
    .sidebar-info {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.06);
        font-size: 0.85rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Model Loading (Cached)
# ──────────────────────────────────────────────

@st.cache_resource
def load_model(checkpoint_path: str = None):
    """Load the trained model. Falls back to untrained model if no checkpoint."""
    device = torch.device("cpu")  # Use CPU for Streamlit inference
    model = AIForensicModel()

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        st.sidebar.success(f"✓ Loaded checkpoint")
    else:
        st.sidebar.warning("⚠ No checkpoint found — using untrained model")

    model.to(device)
    model.eval()
    return model, device


# ──────────────────────────────────────────────
# Analysis Pipeline
# ──────────────────────────────────────────────

def run_analysis(model, image: Image.Image, device):
    """Run the full forensic analysis pipeline on a single image."""
    transform = get_eval_transforms()
    pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(pixel_values)

    probs_a = torch.softmax(outputs["logits_a"][0], dim=0).numpy()
    probs_b = torch.softmax(outputs["logits_b"][0], dim=0).numpy()

    # Attention Rollout
    attn_rollout = compute_attention_rollout(outputs["attentions"])
    heatmap = attention_rollout_to_heatmap(attn_rollout)

    # Saliency maps
    pixel_values_a = transform(image.convert("RGB")).unsqueeze(0).to(device)
    saliency_a = compute_saliency_map(model, pixel_values_a, target_head="a")

    pixel_values_b = transform(image.convert("RGB")).unsqueeze(0).to(device)
    saliency_b = compute_saliency_map(model, pixel_values_b, target_head="b")

    return {
        "probs_a": probs_a,
        "probs_b": probs_b,
        "heatmap": heatmap,
        "saliency_a": saliency_a,
        "saliency_b": saliency_b,
    }


# ──────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 AI-Trust Forensic Suite</h1>
        <p>Detect AI-generated images • Identify the generator model • Visualize suspicious regions</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        checkpoint_path = st.text_input(
            "Model Checkpoint Path",
            value="checkpoints/best_model.pt",
            help="Path to the trained model .pt file",
        )

        st.markdown("### 🔍 Overlay Settings")
        heatmap_alpha = st.slider("Heatmap Opacity", 0.0, 1.0, 0.5, 0.05)
        colormap = st.selectbox(
            "Colormap",
            ["jet", "inferno", "magma", "plasma", "viridis", "hot"],
            index=0,
        )

        st.markdown("""
        <div class="sidebar-info">
            <strong>About</strong><br>
            DATS 6499 Capstone Project<br>
            Mohammed Hussain Nathani<br>
            George Washington University
        </div>
        """, unsafe_allow_html=True)

    # Load model
    model, device = load_model(checkpoint_path)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image for forensic analysis",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Drag & drop or click to upload. Supports JPG, PNG, WebP.",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Show image
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col_info:
            st.markdown(f"""
            **Filename:** `{uploaded_file.name}`
            **Size:** {uploaded_file.size / 1024:.1f} KB
            **Dimensions:** {image.size[0]} × {image.size[1]}
            **Mode:** {image.mode}
            """)

        # Run analysis
        with st.spinner("🔬 Analyzing image..."):
            results = run_analysis(model, image, device)

        pred_a = results["probs_a"].argmax()
        pred_b = results["probs_b"].argmax()
        conf_a = results["probs_a"][pred_a] * 100
        conf_b = results["probs_b"][pred_b] * 100

        # ── Verdict ──
        verdict_class = "verdict-authentic" if pred_a == 0 else "verdict-synthetic"
        verdict_emoji = "✅" if pred_a == 0 else "🚨"
        st.markdown(f"""
        <div class="verdict-card {verdict_class}">
            <h2 style="margin:0; color: white;">{verdict_emoji} Verdict: {LABEL_A_NAMES[pred_a]}</h2>
            <p style="color: rgba(255,255,255,0.7); margin-top: 0.5rem;">
                Confidence: <strong>{conf_a:.1f}%</strong> │
                Predicted Source: <strong>{LABEL_B_NAMES[pred_b]}</strong> ({conf_b:.1f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics Row ──
        st.markdown('<div class="section-header">📊 Classification Probabilities</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Head A — Authenticity Detection**")
            for i, (label, prob) in enumerate(zip(LABEL_A_NAMES.values(), results["probs_a"])):
                st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")

        with col2:
            st.markdown("**Head B — Generator Attribution**")
            # Sort by probability (descending)
            sorted_indices = np.argsort(results["probs_b"])[::-1]
            for i in sorted_indices:
                label = LABEL_B_NAMES[i]
                prob = results["probs_b"][i]
                st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")

        # ── Heatmaps ──
        st.markdown('<div class="section-header">🔥 Forensic Heatmaps — Suspicious Regions</div>', unsafe_allow_html=True)

        img_display = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

        hm_col1, hm_col2, hm_col3 = st.columns(3)

        with hm_col1:
            overlay_attn = create_heatmap_overlay(img_display, results["heatmap"], alpha=heatmap_alpha, colormap=colormap)
            st.image(overlay_attn, caption="Attention Rollout", use_container_width=True, clamp=True)
            st.caption("Highlights regions the model focuses on globally across all transformer layers.")

        with hm_col2:
            overlay_a = create_heatmap_overlay(img_display, results["saliency_a"], alpha=heatmap_alpha, colormap=colormap)
            st.image(overlay_a, caption=f"Saliency: {LABEL_A_NAMES[pred_a]}", use_container_width=True, clamp=True)
            st.caption("Pixel regions most influential for the authenticity decision.")

        with hm_col3:
            overlay_b = create_heatmap_overlay(img_display, results["saliency_b"], alpha=heatmap_alpha, colormap=colormap)
            st.image(overlay_b, caption=f"Saliency: {LABEL_B_NAMES[pred_b]}", use_container_width=True, clamp=True)
            st.caption("Pixel regions most influential for the generator attribution.")

    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: rgba(255,255,255,0.4);">
            <h2 style="font-size: 3rem; margin-bottom: 1rem;">📤</h2>
            <h3>Upload an image to begin forensic analysis</h3>
            <p>Supports JPG, PNG, WebP • Max recommended size: 10MB</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
