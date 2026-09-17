"""
data_loader.py
--------------
DataLoaderModule: Handles EEE domain-specific dataset generation, splitting, PyTorch DataLoader
construction, and EDA for the Solar PV Panel Fault Dataset.
"""

import os
import json
import numpy as np


def load_config(config_path: str = "config/hyperparameters.json") -> dict:
    """Load hyperparameters JSON configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_mnist_dataloaders(config: dict):
    """
    Prepare PyTorch DataLoaders for Solar PV Panel Fault Dataset (EEE Domain).

    Parameters
    ----------
    config : dict
        Configuration dictionary containing hyperparameters.

    Returns
    -------
    tuple (train_loader, val_loader, test_loader)
        PyTorch DataLoaders for train, validation, and test sets.
    """
    import torch
    from torch.utils.data import DataLoader, random_split
    from data_augmentation import get_transforms
    from eee_dataset import SolarPVFaultDataset

    raw_dir = config["dataset"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    train_transform, eval_transform = get_transforms(config)
    seed = config["training"].get("random_seed", 42)

    # 6,000 full train/val pool, 1,000 test set
    full_train_ds = SolarPVFaultDataset(num_samples=6000, seed=seed, transform=train_transform)
    full_eval_ds  = SolarPVFaultDataset(num_samples=6000, seed=seed, transform=eval_transform)
    test_ds       = SolarPVFaultDataset(num_samples=1000, seed=seed + 1, transform=eval_transform)

    generator = torch.Generator().manual_seed(seed)
    train_size = int(len(full_train_ds) * config["dataset"].get("train_val_split_ratio", 0.8333333333333334))
    val_size   = len(full_train_ds) - train_size

    train_subset, _ = random_split(full_train_ds, [train_size, val_size], generator=generator)
    _, val_subset   = random_split(full_eval_ds,  [train_size, val_size], generator=generator)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,       batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[DataLoader] Loaded {config['dataset']['name']} ({config['dataset']['application']}):")
    print(f"  Training samples   : {len(train_subset):,}")
    print(f"  Validation samples : {len(val_subset):,}")
    print(f"  Test samples       : {len(test_ds):,}")
    print(f"  Number of classes  : {config['dataset']['num_classes']} {config['dataset']['class_names']}")
    print(f"  Batch size         : {batch_size}")

    return train_loader, val_loader, test_loader


def explore_dataset_summary(train_loader, test_loader) -> None:
    """Print class distribution and sample batch properties."""
    print("\n-- Dataset Exploration (Solar PV Faults - EEE Domain) ----------------")
    for images, labels in train_loader:
        print(f"  Sample batch tensor shape : {list(images.shape)}")
        print(f"  Data type                 : {images.dtype}")
        print(f"  Batch pixel range [min,max]: [{images.min().item():.3f}, {images.max().item():.3f}]")
        print(f"  Sample label batch        : {labels[:10].tolist()}")
        break
    print("----------------------------------------------------------------------\n")


if __name__ == "__main__":
    cfg = load_config()
    try:
        t_loader, v_loader, te_loader = prepare_mnist_dataloaders(cfg)
        explore_dataset_summary(t_loader, te_loader)
    except ModuleNotFoundError as e:
        print(f"[DataLoader] Deep learning package pending: {e}")
