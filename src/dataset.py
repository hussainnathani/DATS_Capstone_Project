"""
dataset.py — Defactify Dataset Loader
=======================================
HuggingFace datasets-based loader for the Defactify Image Dataset.
Returns (pixel_values, label_a, label_b) for multi-task training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image

from src.utils import (
    DATASET_NAME,
    BATCH_SIZE,
    NUM_WORKERS,
    get_train_transforms,
    get_eval_transforms,
)


class DefactifyDataset(Dataset):
    """
    PyTorch Dataset wrapper around the HuggingFace Defactify dataset.

    Each sample returns:
        pixel_values: transformed image tensor (3, 224, 224)
        label_a: int — binary label (0=Real, 1=AI-Generated)
        label_b: int — generator class (0=Real, 1=SD21, 2=SDXL, 3=SD3, 4=DALLE3, 5=Midjourney)
    """

    def __init__(self, split: str = "train", transform=None, max_samples: int = None):
        """
        Args:
            split: One of "train", "validation", or "test"
            transform: torchvision transforms to apply to images
            max_samples: If set, limits the dataset size (useful for debugging)
        """
        print(f"Loading Defactify dataset — split: {split}...")
        self.hf_dataset = load_dataset(DATASET_NAME, split=split)

        if max_samples is not None:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))
            print(f"  → Limited to {len(self.hf_dataset)} samples")

        self.transform = transform
        print(f"  → Loaded {len(self.hf_dataset)} samples")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]

        # Load image — HuggingFace returns PIL Image
        image = sample["Image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)

        # Convert to RGB (handle grayscale or RGBA)
        image = image.convert("RGB")

        # Apply transforms
        if self.transform is not None:
            pixel_values = self.transform(image)
        else:
            pixel_values = get_eval_transforms()(image)

        label_a = torch.tensor(sample["Label_A"], dtype=torch.long)
        label_b = torch.tensor(sample["Label_B"], dtype=torch.long)

        return pixel_values, label_a, label_b


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    max_samples: int = None,
):
    """
    Create train, validation, and test DataLoaders.

    Args:
        batch_size: Batch size for all loaders
        num_workers: Number of data loading workers
        max_samples: If set, limits each split (useful for debugging)

    Returns:
        dict: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    train_dataset = DefactifyDataset(
        split="train",
        transform=get_train_transforms(),
        max_samples=max_samples,
    )
    val_dataset = DefactifyDataset(
        split="validation",
        transform=get_eval_transforms(),
        max_samples=max_samples,
    )
    test_dataset = DefactifyDataset(
        split="test",
        transform=get_eval_transforms(),
        max_samples=max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\n📊 DataLoader Summary:")
    print(f"  Train:      {len(train_dataset):>6} samples → {len(train_loader):>4} batches")
    print(f"  Validation: {len(val_dataset):>6} samples → {len(val_loader):>4} batches")
    print(f"  Test:       {len(test_dataset):>6} samples → {len(test_loader):>4} batches")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
