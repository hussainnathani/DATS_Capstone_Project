"""
utils.py — Shared Utilities & Configuration
=============================================
Central configuration, image transforms, device helpers, and label mappings
for the AI-Trust Forensic Suite.
"""

import torch
from torchvision import transforms

# ──────────────────────────────────────────────
# Configuration Constants
# ──────────────────────────────────────────────

# HuggingFace model identifier (free, open-source)
VIT_MODEL_NAME = "google/vit-base-patch16-224"

# HuggingFace dataset identifier
DATASET_NAME = "Rajarshi-Roy-research/Defactify_Image_Dataset"

# Image configuration
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# Training defaults
LEARNING_RATE = 2e-5
NUM_EPOCHS = 15
WEIGHT_DECAY = 0.05   # L2 regularization (increased from 0.01)
L1_LAMBDA = 1e-5       # L1 regularization strength
WARMUP_RATIO = 0.1

# Multi-task loss weights
ALPHA = 2.0  # Weight for Head A (binary detection) — prioritize fixing Real detection
BETA = 1.0   # Weight for Head B (generator attribution)

# Class weights for Head A (Real is 5x underrepresented)
CLASS_WEIGHTS_A = [8.0, 1.0]  # [Real, AI-Generated] — push harder on Real recall

# Number of classes
NUM_CLASSES_A = 2  # Real vs AI-Generated
NUM_CLASSES_B = 6  # Real, SD21, SDXL, SD3, DALLE3, Midjourney

# Freeze strategy: number of ViT encoder layers to FREEZE (out of 12)
NUM_FROZEN_LAYERS = 6  # Unfreeze 6 layers (was 8) for better generator fingerprints

# ──────────────────────────────────────────────
# Label Mappings
# ──────────────────────────────────────────────

LABEL_A_NAMES = {
    0: "Authentic (Real)",
    1: "AI-Generated (Synthetic)",
}

LABEL_B_NAMES = {
    0: "Real",
    1: "SD 2.1",
    2: "SDXL",
    3: "SD 3",
    4: "DALL·E 3",
    5: "Midjourney",
}

# ──────────────────────────────────────────────
# Image Transforms
# ──────────────────────────────────────────────

def get_train_transforms():
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms():
    """Evaluation/inference transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ──────────────────────────────────────────────
# Device Helper
# ──────────────────────────────────────────────

def get_device():
    """Select the best available device: TPU > CUDA > MPS > CPU."""
    # Check for TPU (Google Colab with TPU runtime)
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"✓ Using TPU device: {device}")
        return device
    except ImportError:
        pass

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple MPS device")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU — training will be slow")
    return device


def count_parameters(model, trainable_only=True):
    """Count total or trainable parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
