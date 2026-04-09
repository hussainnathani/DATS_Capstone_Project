"""
evaluate.py — Evaluation Metrics for Multi-Task Model
=======================================================
Computes per-head accuracy, precision, recall, F1 score,
confusion matrix, and full classification reports.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.utils import LABEL_A_NAMES, LABEL_B_NAMES


def compute_metrics(y_true, y_pred, label_names=None):
    """
    Compute classification metrics for a single head.

    Returns:
        dict with accuracy, precision, recall, f1 (macro)
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Run full evaluation on a DataLoader.

    Returns:
        dict with metrics for Head A and Head B, plus raw predictions
    """
    model.eval()
    all_labels_a, all_preds_a = [], []
    all_labels_b, all_preds_b = [], []

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for pixel_values, labels_a, labels_b in pbar:
        pixel_values = pixel_values.to(device)

        outputs = model(pixel_values)
        preds_a = torch.argmax(outputs["logits_a"], dim=1).cpu().numpy()
        preds_b = torch.argmax(outputs["logits_b"], dim=1).cpu().numpy()

        all_labels_a.extend(labels_a.numpy())
        all_preds_a.extend(preds_a)
        all_labels_b.extend(labels_b.numpy())
        all_preds_b.extend(preds_b)

    all_labels_a = np.array(all_labels_a)
    all_preds_a = np.array(all_preds_a)
    all_labels_b = np.array(all_labels_b)
    all_preds_b = np.array(all_preds_b)

    metrics_a = compute_metrics(all_labels_a, all_preds_a)
    metrics_b = compute_metrics(all_labels_b, all_preds_b)

    return {
        "head_a": metrics_a,
        "head_b": metrics_b,
        "labels_a": all_labels_a,
        "preds_a": all_preds_a,
        "labels_b": all_labels_b,
        "preds_b": all_preds_b,
    }


def print_classification_reports(results):
    """Print detailed classification reports for both heads."""
    print("\n" + "=" * 60)
    print("HEAD A — Binary Detection (Authentic vs Synthetic)")
    print("=" * 60)
    target_names_a = [LABEL_A_NAMES[i] for i in sorted(LABEL_A_NAMES.keys())]
    print(classification_report(
        results["labels_a"], results["preds_a"],
        target_names=target_names_a, digits=4
    ))

    print("\n" + "=" * 60)
    print("HEAD B — Generator Attribution")
    print("=" * 60)
    target_names_b = [LABEL_B_NAMES[i] for i in sorted(LABEL_B_NAMES.keys())]
    print(classification_report(
        results["labels_b"], results["preds_b"],
        target_names=target_names_b, digits=4
    ))


def plot_confusion_matrices(results, save_path=None):
    """Plot confusion matrices for both heads side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Head A
    cm_a = confusion_matrix(results["labels_a"], results["preds_a"])
    target_names_a = [LABEL_A_NAMES[i] for i in sorted(LABEL_A_NAMES.keys())]
    sns.heatmap(
        cm_a, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names_a, yticklabels=target_names_a,
        ax=axes[0],
    )
    axes[0].set_title("Head A — Binary Detection", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Head B
    cm_b = confusion_matrix(results["labels_b"], results["preds_b"])
    target_names_b = [LABEL_B_NAMES[i] for i in sorted(LABEL_B_NAMES.keys())]
    sns.heatmap(
        cm_b, annot=True, fmt="d", cmap="Oranges",
        xticklabels=target_names_b, yticklabels=target_names_b,
        ax=axes[1],
    )
    axes[1].set_title("Head B — Generator Attribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Confusion matrices saved to {save_path}")
    plt.show()


def plot_training_history(history, save_path=None):
    """Plot training curves from history list."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    val_acc_a = [h["val"]["acc_a"] * 100 for h in history]
    val_acc_b = [h["val"]["acc_b"] * 100 for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, train_loss, "o-", label="Train Loss", color="#2196F3")
    axes[0].plot(epochs, val_loss, "o-", label="Val Loss", color="#F44336")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, val_acc_a, "o-", label="Val Acc (Detection)", color="#4CAF50")
    axes[1].plot(epochs, val_acc_b, "o-", label="Val Acc (Attribution)", color="#FF9800")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Validation Accuracy", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Training curves saved to {save_path}")
    plt.show()
