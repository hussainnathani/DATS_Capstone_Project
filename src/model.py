"""
model.py — Multi-Task ViT for AI Forensics
=============================================
Shared Vision Transformer backbone with two classification heads:
  Head A: Binary detection (Authentic vs Synthetic)
  Head B: Generator attribution (Real, SD21, SDXL, SD3, DALLE3, Midjourney)
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

from src.utils import (
    VIT_MODEL_NAME,
    NUM_CLASSES_A,
    NUM_CLASSES_B,
    NUM_FROZEN_LAYERS,
    count_parameters,
)


class AIForensicModel(nn.Module):
    """
    Multi-Task Vision Transformer for AI-generated image forensics.

    Architecture:
        ViT-Base/16 backbone (pretrained on ImageNet-21k)
        ├── Head A: Binary Detection  (Authentic vs Synthetic)
        └── Head B: Generator Attribution (6-class)

    The [CLS] token embedding from the ViT is shared between both heads.
    """

    def __init__(
        self,
        model_name: str = VIT_MODEL_NAME,
        num_classes_a: int = NUM_CLASSES_A,
        num_classes_b: int = NUM_CLASSES_B,
        num_frozen_layers: int = NUM_FROZEN_LAYERS,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # ── Shared backbone ──
        self.backbone = ViTModel.from_pretrained(
            model_name,
            add_pooling_layer=True,        # Use built-in [CLS] pooler
            output_attentions=True,        # Enable attention output for XAI
        )
        hidden_size = self.backbone.config.hidden_size  # 768 for ViT-Base

        # ── Freeze early layers ──
        self._freeze_layers(num_frozen_layers)

        # ── Head A: Binary Detection ──
        self.head_a = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes_a),
        )

        # ── Head B: Generator Attribution ──
        self.head_b = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes_b),
        )

        # Initialize classification heads
        self._init_heads()

    def _freeze_layers(self, num_frozen: int):
        """Freeze the embeddings and the first `num_frozen` encoder layers."""
        # Freeze patch + position embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified number of encoder layers
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < num_frozen:
                for param in layer.parameters():
                    param.requires_grad = False

    def _init_heads(self):
        """Xavier initialization for classification head linear layers."""
        for head in [self.head_a, self.head_b]:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, pixel_values: torch.Tensor):
        """
        Forward pass through backbone + both heads.

        Args:
            pixel_values: Tensor of shape (B, 3, 224, 224)

        Returns:
            dict with keys:
                "logits_a": (B, num_classes_a) — binary detection logits
                "logits_b": (B, num_classes_b) — generator attribution logits
                "attentions": tuple of attention weight tensors (for XAI)
        """
        outputs = self.backbone(pixel_values=pixel_values)

        # Pooled [CLS] token representation
        pooled = outputs.pooler_output  # (B, 768)

        logits_a = self.head_a(pooled)  # (B, 2)
        logits_b = self.head_b(pooled)  # (B, 6)

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "attentions": outputs.attentions,  # Tuple of (B, heads, seq, seq)
        }

    def get_trainable_summary(self):
        """Print a summary of total vs trainable parameters."""
        total = count_parameters(self, trainable_only=False)
        trainable = count_parameters(self, trainable_only=True)
        frozen = total - trainable
        print(f"{'Total Parameters:':<30} {total:>12,}")
        print(f"{'Trainable Parameters:':<30} {trainable:>12,}")
        print(f"{'Frozen Parameters:':<30} {frozen:>12,}")
        print(f"{'Trainable %:':<30} {100 * trainable / total:>11.1f}%")
        return {"total": total, "trainable": trainable, "frozen": frozen}


class MultiTaskLoss(nn.Module):
    """
    Combined cross-entropy loss for both classification heads.

    Loss = alpha * CE(head_a) + beta * CE(head_b)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_a = nn.CrossEntropyLoss()
        self.ce_b = nn.CrossEntropyLoss()

    def forward(self, logits_a, logits_b, labels_a, labels_b):
        """
        Compute combined multi-task loss.

        Returns:
            dict: "loss" (total), "loss_a", "loss_b"
        """
        loss_a = self.ce_a(logits_a, labels_a)
        loss_b = self.ce_b(logits_b, labels_b)
        total = self.alpha * loss_a + self.beta * loss_b
        return {
            "loss": total,
            "loss_a": loss_a,
            "loss_b": loss_b,
        }
