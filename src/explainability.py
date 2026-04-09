"""
explainability.py — XAI: Attention Rollout & Saliency Maps
============================================================
Provides visual explanations for model predictions using:
  1. Attention Rollout — aggregated attention across all ViT layers
  2. Saliency Maps — input gradient-based attribution
Generates heatmap overlays on original images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.utils import get_eval_transforms, IMAGE_SIZE


# ──────────────────────────────────────────────
# Attention Rollout
# ──────────────────────────────────────────────

def compute_attention_rollout(attentions, head_fusion="mean"):
    """
    Compute Attention Rollout from all ViT layer attention matrices.

    This method multiplies the attention matrices across layers to compute
    the total attention flow from the input patches to the final [CLS] token.

    Args:
        attentions: Tuple of attention tensors from ViT, each (B, heads, seq, seq)
        head_fusion: How to combine attention heads — "mean", "max", or "min"

    Returns:
        rollout: numpy array of shape (num_patches,) — attention map for [CLS] token
    """
    # Stack and take first sample in batch
    result = None
    for attention in attentions:
        # attention shape: (B, num_heads, seq_len, seq_len)
        attn = attention[0]  # First sample: (num_heads, seq_len, seq_len)

        # Fuse attention heads
        if head_fusion == "mean":
            attn_fused = attn.mean(dim=0)  # (seq_len, seq_len)
        elif head_fusion == "max":
            attn_fused = attn.max(dim=0).values
        elif head_fusion == "min":
            attn_fused = attn.min(dim=0).values
        else:
            raise ValueError(f"Unknown head_fusion: {head_fusion}")

        # Add identity matrix (residual connection)
        I = torch.eye(attn_fused.size(0), device=attn_fused.device)
        attn_fused = 0.5 * attn_fused + 0.5 * I

        # Renormalize rows
        attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True)

        # Multiply through layers
        if result is None:
            result = attn_fused
        else:
            result = torch.matmul(attn_fused, result)

    # Extract attention to [CLS] token (index 0), excluding [CLS] itself
    cls_attention = result[0, 1:]  # (num_patches,)
    return cls_attention.detach().cpu().numpy()


def attention_rollout_to_heatmap(cls_attention, image_size=IMAGE_SIZE):
    """
    Reshape the 1D patch-level attention vector into a 2D spatial heatmap.

    Args:
        cls_attention: 1D numpy array of shape (num_patches,)
        image_size: Original image size (default 224)

    Returns:
        heatmap: 2D numpy array of shape (image_size, image_size)
    """
    # ViT-Base/16 → 14×14 patches
    patch_grid = int(np.sqrt(len(cls_attention)))
    heatmap = cls_attention.reshape(patch_grid, patch_grid)

    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Resize to original image dimensions
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_pil = heatmap_pil.resize((image_size, image_size), Image.BICUBIC)
    return np.array(heatmap_pil) / 255.0


# ──────────────────────────────────────────────
# Saliency Map (Input Gradient)
# ──────────────────────────────────────────────

def compute_saliency_map(model, pixel_values, target_head="a", target_class=None):
    """
    Compute a gradient-based saliency map for the input image.

    Args:
        model: The AIForensicModel
        pixel_values: Input tensor of shape (1, 3, 224, 224), requires_grad=True
        target_head: "a" for detection head, "b" for attribution head
        target_class: Class index to compute gradient for. If None, uses predicted class.

    Returns:
        saliency: 2D numpy array of shape (224, 224)
    """
    model.eval()
    pixel_values = pixel_values.clone().detach().requires_grad_(True)

    outputs = model(pixel_values)

    if target_head == "a":
        logits = outputs["logits_a"]
    else:
        logits = outputs["logits_b"]

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    # Backpropagate w.r.t. the target class score
    score = logits[0, target_class]
    score.backward()

    # Take max absolute gradient across color channels
    saliency = pixel_values.grad.data[0].abs().max(dim=0).values  # (224, 224)
    saliency = saliency.cpu().numpy()

    # Normalize to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def create_heatmap_overlay(original_image, heatmap, alpha=0.5, colormap="jet"):
    """
    Overlay a heatmap on the original image.

    Args:
        original_image: PIL Image or numpy array
        heatmap: 2D numpy array of shape (H, W) in [0, 1]
        alpha: Blending factor (0 = original only, 1 = heatmap only)
        colormap: Matplotlib colormap name

    Returns:
        overlay: numpy array of shape (H, W, 3)
    """
    if isinstance(original_image, Image.Image):
        original_image = original_image.resize((heatmap.shape[1], heatmap.shape[0]))
        original_image = np.array(original_image) / 255.0

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel

    # Blend
    overlay = (1 - alpha) * original_image + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    return overlay


def visualize_forensic_analysis(
    model,
    image: Image.Image,
    device,
    label_a_names: dict = None,
    label_b_names: dict = None,
    save_path: str = None,
):
    """
    Full forensic visualization pipeline for a single image.

    Displays:
        - Original image
        - Attention Rollout heatmap
        - Saliency map for Head A
        - Saliency map for Head B
        - Prediction scores

    Args:
        model: Trained AIForensicModel
        image: PIL Image to analyze
        device: torch device
        label_a_names: dict mapping class indices to names for Head A
        label_b_names: dict mapping class indices to names for Head B
        save_path: If provided, save the figure to this path
    """
    from src.utils import LABEL_A_NAMES, LABEL_B_NAMES

    if label_a_names is None:
        label_a_names = LABEL_A_NAMES
    if label_b_names is None:
        label_b_names = LABEL_B_NAMES

    # Prepare image
    transform = get_eval_transforms()
    pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(device)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values)

    probs_a = torch.softmax(outputs["logits_a"][0], dim=0).cpu().numpy()
    probs_b = torch.softmax(outputs["logits_b"][0], dim=0).cpu().numpy()
    pred_a = probs_a.argmax()
    pred_b = probs_b.argmax()

    # XAI maps
    attn_rollout = compute_attention_rollout(outputs["attentions"])
    heatmap_attn = attention_rollout_to_heatmap(attn_rollout)

    pixel_values_grad = transform(image.convert("RGB")).unsqueeze(0).to(device)
    saliency_a = compute_saliency_map(model, pixel_values_grad, target_head="a")
    pixel_values_grad = transform(image.convert("RGB")).unsqueeze(0).to(device)
    saliency_b = compute_saliency_map(model, pixel_values_grad, target_head="b")

    # Resize original for display
    img_display = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img_display)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    overlay_attn = create_heatmap_overlay(img_display, heatmap_attn)
    axes[1].imshow(overlay_attn)
    axes[1].set_title("Attention Rollout", fontweight="bold")
    axes[1].axis("off")

    overlay_sal_a = create_heatmap_overlay(img_display, saliency_a)
    axes[2].imshow(overlay_sal_a)
    axes[2].set_title(f"Saliency: {label_a_names[pred_a]}\n({probs_a[pred_a]*100:.1f}%)", fontweight="bold")
    axes[2].axis("off")

    overlay_sal_b = create_heatmap_overlay(img_display, saliency_b)
    axes[3].imshow(overlay_sal_b)
    axes[3].set_title(f"Source: {label_b_names[pred_b]}\n({probs_b[pred_b]*100:.1f}%)", fontweight="bold")
    axes[3].axis("off")

    plt.suptitle(
        f"Verdict: {label_a_names[pred_a]} ({probs_a[pred_a]*100:.1f}%) │ "
        f"Source: {label_b_names[pred_b]} ({probs_b[pred_b]*100:.1f}%)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Forensic analysis saved to {save_path}")
    plt.show()

    return {
        "probs_a": probs_a,
        "probs_b": probs_b,
        "pred_a": pred_a,
        "pred_b": pred_b,
        "heatmap_attn": heatmap_attn,
        "saliency_a": saliency_a,
        "saliency_b": saliency_b,
    }
