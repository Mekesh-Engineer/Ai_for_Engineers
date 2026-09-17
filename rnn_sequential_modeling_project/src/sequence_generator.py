# -*- coding: utf-8 -*-
"""
sequence_generator.py
---------------------
SequenceGeneratorModule: Sliding lookback window sequence generator and
temporal train/validation/test dataset partitioning for PyTorch Vanilla RNN.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def create_sequences(data: np.ndarray, lookback: int = 30, lookahead: int = 1) -> tuple:
    """
    Generate sliding window sequences from a 1D or 2D time series array.

    Parameters
    ----------
    data : np.ndarray
        Scaled 1D or 2D array of shape (N,) or (N, features).
    lookback : int
        Number of historical timesteps to use as input sequence (default: 30).
    lookahead : int
        Number of future steps to predict (default: 1).

    Returns
    -------
    tuple (X, y)
        X : np.ndarray of shape (num_samples, lookback, features)
        y : np.ndarray of shape (num_samples, lookahead)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    num_timestamps = len(data)
    X, y = [], []

    for i in range(num_timestamps - lookback - lookahead + 1):
        window_x = data[i : i + lookback, :]
        window_y = data[i + lookback : i + lookback + lookahead, 0]
        X.append(window_x)
        y.append(window_y)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)

    return X_arr, y_arr


def split_time_series_sequences(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8, val_ratio: float = 0.1) -> tuple:
    """
    Perform temporal sequential train/validation/test split without shuffling
    to preserve time-series ordering and prevent temporal data leakage.

    Parameters
    ----------
    X : np.ndarray
        Feature sequence array of shape (N, lookback, features).
    y : np.ndarray
        Target array of shape (N, 1).
    train_ratio : float
        Fraction of data for training (default: 0.8).
    val_ratio : float
        Fraction of data for validation (default: 0.1).

    Returns
    -------
    tuple (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    total_samples = len(X)
    train_end = int(total_samples * train_ratio)
    val_end   = train_end + int(total_samples * val_ratio)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    print(f"[SequenceGenerator] Created sequential dataset splits:")
    print(f"  - Total Sequences : {total_samples:,}")
    print(f"  - Train Set       : {len(X_train):,} samples ({train_ratio*100:.1f}%) | Shape: {X_train.shape}")
    print(f"  - Val Set         : {len(X_val):,} samples ({val_ratio*100:.1f}%)   | Shape: {X_val.shape}")
    print(f"  - Test Set        : {len(X_test):,} samples ({(1 - train_ratio - val_ratio)*100:.1f}%)  | Shape: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_pytorch_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int = 32) -> tuple:
    """Wrap numpy arrays into PyTorch TensorDatasets and DataLoaders."""
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

    # Note: shuffle=True for training batch gradient descent, shuffle=False for val & test
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dummy_series = np.linspace(0, 100, 1000)
    X, y = create_sequences(dummy_series, lookback=30)
    X_tr, y_tr, X_v, y_v, X_te, y_te = split_time_series_sequences(X, y)
    print(f"[SequenceGenerator Test] X_tr shape: {X_tr.shape}, y_tr shape: {y_tr.shape}")
