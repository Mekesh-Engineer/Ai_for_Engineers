"""
training.py
-----------
TrainingModule: Executes PyTorch CNN training loop, handles optimization,
validation monitoring, early stopping, and best checkpoint saving.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed: int = 42) -> None:
    """Set random seed across PyTorch, NumPy, and random for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, save_path: str = "models/cnn_model.pt"):
        self.patience  = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter   = 0
        self.early_stop = False

    def check(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: torch.device) -> tuple:
    """Train model for a single epoch."""
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds      = torch.max(outputs, 1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def evaluate_one_epoch(model: nn.Module, loader, criterion, device: torch.device) -> tuple:
    """Evaluate model for a single epoch."""
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds      = torch.max(outputs, 1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def train_model(model: nn.Module, train_loader, val_loader, config: dict) -> dict:
    """
    Complete model training pipeline with validation and early stopping.

    Parameters
    ----------
    model : nn.Module
        PyTorch CNN model instance.
    train_loader : DataLoader
        Training set loader.
    val_loader : DataLoader
        Validation set loader.
    config : dict
        Hyperparameters configuration.

    Returns
    -------
    dict
        Training history dictionary containing loss and accuracy records.
    """
    t_cfg = config["training"]
    set_seed(t_cfg.get("random_seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using compute device: {device}")
    model.to(device)

    # Optimizer & Loss Function
    lr          = t_cfg.get("learning_rate", 0.001)
    weight_decay= t_cfg.get("weight_decay", 1e-4)
    optimizer   = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion   = nn.CrossEntropyLoss()

    epochs      = t_cfg.get("epochs", 30)
    patience    = t_cfg.get("early_stopping_patience", 5)
    model_path  = config["output"]["model_path"]

    early_stopping = EarlyStopping(patience=patience, save_path=model_path)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [],   "val_acc": [],
        "epoch_times": []
    }

    print("\n── Starting CNN Model Training ──────────────────────────────")
    print(f"  Epochs: {epochs} | Batch Size: {t_cfg['batch_size']} | LR: {lr} | Patience: {patience}")
    print("─────────────────────────────────────────────────────────────")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc= evaluate_one_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(elapsed)

        saved_flag = " ★ [Saved Best]" if early_stopping.check(val_loss, model) else ""

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s]  "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:6.2f}%  "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}%{saved_flag}",
            flush=True
        )

        if early_stopping.early_stop:
            print(f"\n[Training] Early stopping triggered at epoch {epoch}. Restoring best checkpoint.")
            break

    print("─────────────────────────────────────────────────────────────\n")
    # Load best model weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    return history


if __name__ == "__main__":
    print("[Training] Module compiled successfully.")
