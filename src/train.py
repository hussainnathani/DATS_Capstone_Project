"""
train.py — Multi-Task Training Pipeline
==========================================
Training loop for the AIForensicModel with multi-task loss,
layer freezing, learning rate scheduling, and checkpoint saving.
"""

import os
import time
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.model import AIForensicModel, MultiTaskLoss
from src.dataset import get_dataloaders
from src.evaluate import compute_metrics
from src.utils import (
    get_device,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    ALPHA,
    BETA,
    BATCH_SIZE,
)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for a single epoch."""
    model.train()
    running_loss = 0.0
    running_loss_a = 0.0
    running_loss_b = 0.0
    correct_a = 0
    correct_b = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for pixel_values, labels_a, labels_b in pbar:
        pixel_values = pixel_values.to(device)
        labels_a = labels_a.to(device)
        labels_b = labels_b.to(device)

        # Forward pass
        outputs = model(pixel_values)
        losses = criterion(outputs["logits_a"], outputs["logits_b"], labels_a, labels_b)

        # Backward pass
        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        batch_size = pixel_values.size(0)
        running_loss += losses["loss"].item() * batch_size
        running_loss_a += losses["loss_a"].item() * batch_size
        running_loss_b += losses["loss_b"].item() * batch_size

        _, preds_a = torch.max(outputs["logits_a"], 1)
        _, preds_b = torch.max(outputs["logits_b"], 1)
        correct_a += (preds_a == labels_a).sum().item()
        correct_b += (preds_b == labels_b).sum().item()
        total += batch_size

        pbar.set_postfix({
            "loss": f"{losses['loss'].item():.4f}",
            "acc_a": f"{100 * correct_a / total:.1f}%",
            "acc_b": f"{100 * correct_b / total:.1f}%",
        })

    return {
        "loss": running_loss / total,
        "loss_a": running_loss_a / total,
        "loss_b": running_loss_b / total,
        "acc_a": correct_a / total,
        "acc_b": correct_b / total,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    """Validate the model on the validation set."""
    model.eval()
    running_loss = 0.0
    correct_a = 0
    correct_b = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]  ", leave=False)
    for pixel_values, labels_a, labels_b in pbar:
        pixel_values = pixel_values.to(device)
        labels_a = labels_a.to(device)
        labels_b = labels_b.to(device)

        outputs = model(pixel_values)
        losses = criterion(outputs["logits_a"], outputs["logits_b"], labels_a, labels_b)

        batch_size = pixel_values.size(0)
        running_loss += losses["loss"].item() * batch_size

        _, preds_a = torch.max(outputs["logits_a"], 1)
        _, preds_b = torch.max(outputs["logits_b"], 1)
        correct_a += (preds_a == labels_a).sum().item()
        correct_b += (preds_b == labels_b).sum().item()
        total += batch_size

    return {
        "loss": running_loss / total,
        "acc_a": correct_a / total,
        "acc_b": correct_b / total,
    }


def train(
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    alpha: float = ALPHA,
    beta: float = BETA,
    checkpoint_dir: str = "checkpoints",
    max_samples: int = None,
):
    """
    Full training pipeline.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        alpha: Weight for Head A loss
        beta: Weight for Head B loss
        checkpoint_dir: Directory to save checkpoints
        max_samples: Limit dataset size (for debugging)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = get_device()

    # ── Data ──
    print("\n📦 Loading datasets...")
    loaders = get_dataloaders(batch_size=batch_size, max_samples=max_samples)

    # ── Model ──
    print("\n🧠 Initializing model...")
    model = AIForensicModel().to(device)
    model.get_trainable_summary()

    # ── Optimizer & Scheduler ──
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    criterion = MultiTaskLoss(alpha=alpha, beta=beta)

    # ── Training Loop ──
    print(f"\n🚀 Starting training for {num_epochs} epochs\n")
    best_val_acc = 0.0
    history = []

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, epoch
        )
        # Validate
        val_metrics = validate(
            model, loaders["val"], criterion, device, epoch
        )
        scheduler.step()

        elapsed = time.time() - start_time

        # Log
        print(
            f"Epoch {epoch+1}/{num_epochs} ({elapsed:.0f}s) │ "
            f"Train Loss: {train_metrics['loss']:.4f} │ "
            f"Val Loss: {val_metrics['loss']:.4f} │ "
            f"Val Acc A: {100*val_metrics['acc_a']:.1f}% │ "
            f"Val Acc B: {100*val_metrics['acc_b']:.1f}%"
        )

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save best checkpoint
        val_acc_mean = (val_metrics["acc_a"] + val_metrics["acc_b"]) / 2
        if val_acc_mean > best_val_acc:
            best_val_acc = val_acc_mean
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc_a": val_metrics["acc_a"],
                "val_acc_b": val_metrics["acc_b"],
                "history": history,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (avg acc: {100*best_val_acc:.1f}%)")

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "history": history,
    }, final_path)
    print(f"\n✅ Training complete. Best avg val accuracy: {100*best_val_acc:.1f}%")

    return model, history


if __name__ == "__main__":
    # Train on the full dataset
    train(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

