# -*- coding: utf-8 -*-
"""
training.py
-----------
TrainingModule: PyTorch sequence training loop, backpropagation through time (BPTT),
validation monitoring, early stopping, and model state checkpoint saving.
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
) -> dict:
    """
    Train the Vanilla RNN time-series model with Adam optimizer, MSE loss,
    gradient clipping, and early stopping.

    Parameters
    ----------
    model : nn.Module
        StackedRNNRegressor instance.
    train_loader : DataLoader
        PyTorch DataLoader for training sequence batch tensors.
    val_loader : DataLoader
        PyTorch DataLoader for validation sequence batch tensors.
    config : dict
        Hyperparameters configuration dictionary.

    Returns
    -------
    dict
        Training history dictionary containing 'train_loss' and 'val_loss' lists.
    """
    t_cfg = config.get("training", {})
    epochs = t_cfg.get("epochs", 60)
    lr = t_cfg.get("learning_rate", 0.001)
    weight_decay = t_cfg.get("weight_decay", 0.0001)
    patience = t_cfg.get("early_stopping_patience", 12)
    model_path = config.get("output", {}).get("model_path", "models/rnn_model.pt")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": []}

    best_val_loss = float("inf")
    best_model_weights = None
    patience_counter = 0

    print(f"\n[Training] Starting Stacked Vanilla RNN Training on {device} ({epochs} Max Epochs)...")
    print("  Epoch | Train Loss (MSE) | Val Loss (MSE) | Status")
    print("  ------+------------------+----------------+--------------------------")

    for epoch in range(1, epochs + 1):
        # ── Training Phase ─────────────────────────────────────────────────
        model.train()
        running_train_loss = 0.0
        total_train_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()

            # Gradient clipping to prevent exploding gradients in BPTT
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_train_loss += loss.item() * X_batch.size(0)
            total_train_samples += X_batch.size(0)

        epoch_train_loss = running_train_loss / total_train_samples

        # ── Validation Phase ───────────────────────────────────────────────
        model.eval()
        running_val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

                running_val_loss += loss.item() * X_batch.size(0)
                total_val_samples += X_batch.size(0)

        epoch_val_loss = running_val_loss / total_val_samples

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        # ── Checkpoint & Early Stopping ────────────────────────────────────
        status_msg = ""
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, model_path)
            patience_counter = 0
            status_msg = "[OK] Best Checkpoint Saved"
        else:
            patience_counter += 1
            status_msg = f"Patience {patience_counter}/{patience}"

        print(f"   {epoch:02d}/{epochs:02d} |     {epoch_train_loss:8.6f}     |   {epoch_val_loss:8.6f}   | {status_msg}")

        if patience_counter >= patience:
            print(f"\n[Early Stopping] Triggered at epoch {epoch} (Validation loss failed to improve for {patience} consecutive epochs).")
            break

    # Restore best weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"[Training] Restored model weights from best validation loss epoch ({best_val_loss:.6f}).")

    return history


if __name__ == "__main__":
    from model_builder import StackedRNNRegressor
    from sequence_generator import prepare_pytorch_dataloaders
    import numpy as np

    X_dummy = np.random.randn(100, 30, 1).astype(np.float32)
    y_dummy = np.random.randn(100, 1).astype(np.float32)
    t_ld, v_ld, _ = prepare_pytorch_dataloaders(X_dummy[:70], y_dummy[:70], X_dummy[70:85], y_dummy[70:85], X_dummy[85:], y_dummy[85:])

    test_cfg = {"training": {"epochs": 3, "learning_rate": 0.01, "early_stopping_patience": 2}, "output": {"model_path": "models/test_model.pt"}}
    m = StackedRNNRegressor()
    h = train_model(m, t_ld, v_ld, test_cfg)
    print(f"[Training Test] History keys: {list(h.keys())}")
