"""
data_loader.py
==============
DataLoaderModule: Load and explore the Play Tennis categorical dataset.

Functions
---------
load_classification_data(path, drop_cols=None)
    Load CSV, print dimensions, dtypes, and class distribution.
check_data_quality(df, target_col)
    Report missing values, duplicates, and class balance.
augment_data(df, n_samples, random_state)
    Bootstrap-augment small datasets to a target sample count.
"""

import os
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_classification_data(path: str, drop_cols: list = None) -> pd.DataFrame:
    """Load dataset from *path* and return a cleaned DataFrame.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.
    drop_cols : list, optional
        Column names to drop (e.g., index columns like 'Day').

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with optional columns removed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    if drop_cols:
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    print("=" * 60)
    print("DATA LOADING SUMMARY")
    print("=" * 60)
    print(f"  File         : {path}")
    print(f"  Shape        : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\n  Column dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<20} {dtype}")
    print()
    return df


def check_data_quality(df: pd.DataFrame, target_col: str) -> dict:
    """Analyse data quality and class distribution.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Name of the target / label column.

    Returns
    -------
    dict
        Summary statistics (missing counts, duplicate count, class counts).
    """
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    # Missing values
    missing = df.isnull().sum()
    print("\n  Missing values per column:")
    for col, cnt in missing.items():
        flag = "  <- !" if cnt > 0 else ""
        print(f"    {col:<20} {cnt}{flag}")

    total_missing = missing.sum()
    print(f"\n  Total missing cells : {total_missing}")

    # Duplicates
    n_dupes = df.duplicated().sum()
    print(f"  Duplicate rows      : {n_dupes}")
    if n_dupes > 0:
        df.drop_duplicates(inplace=True)
        print("  -> Duplicates removed.")

    # Class distribution
    print(f"\n  Class distribution ({target_col}):")
    vc = df[target_col].value_counts()
    for cls, cnt in vc.items():
        pct = cnt / len(df) * 100
        print(f"    {str(cls):<10} {cnt:>4}  ({pct:.1f}%)")

    # Imbalance warning
    ratio = vc.min() / vc.max()
    if ratio < 0.6:
        print("  [!] Class imbalance detected (minority/majority < 0.6). "
              "Consider class weights.")
    else:
        print("  [OK] Classes are reasonably balanced.")

    print()
    return {
        "missing": missing.to_dict(),
        "duplicates": n_dupes,
        "class_counts": vc.to_dict(),
    }


def augment_data(df: pd.DataFrame, n_samples: int = 140,
                 random_state: int = 42) -> pd.DataFrame:
    """Bootstrap-resample *df* to *n_samples* rows.

    This is used to create a statistically richer dataset from the tiny
    14-record Play Tennis table so that ROC curves and cross-validation
    are meaningful.  The original class proportions are preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (already encoded or raw).
    n_samples : int
        Target number of rows in the output.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame with *n_samples* rows.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(df), size=n_samples, replace=True)
    augmented = df.iloc[idx].reset_index(drop=True)

    print(f"  Data augmented: {len(df)} -> {len(augmented)} rows "
          f"(bootstrap, random_state={random_state})")
    return augmented


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "play_tennis.csv")
    df = load_classification_data(DATA_PATH, drop_cols=["Day"])
    check_data_quality(df, target_col="PlayTennis")
    aug = augment_data(df, n_samples=140)
    print(aug.head())
