"""
data_loader.py
--------------
DataLoaderModule: Handles loading, exploring, and initial inspection
of the Iris dataset (from scikit-learn or a local CSV).
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


def load_config(config_path: str = "config/parameters.json") -> dict:
    """Load project configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def load_iris_data(config: dict = None, save_raw: bool = True) -> pd.DataFrame:
    """
    Load the Iris dataset from scikit-learn and return as a DataFrame.

    Parameters
    ----------
    config : dict, optional
        Project configuration dictionary.
    save_raw : bool
        If True, save the raw dataset to the configured path.

    Returns
    -------
    pd.DataFrame
        Iris dataset with feature columns (no target used in clustering).
    """
    iris = load_iris()
    feature_names = [name.replace(" (cm)", "").replace(" ", "_")
                     for name in iris.feature_names]

    df = pd.DataFrame(iris.data, columns=feature_names)
    # Keep species column for reference / post-hoc validation only
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

    if save_raw and config is not None:
        raw_path = config["data"]["raw_path"]
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        df.to_csv(raw_path, index=False)
        print(f"[DataLoader] Raw dataset saved → {raw_path}")

    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """Print a structured EDA summary of the loaded dataset."""
    print("\n" + "=" * 60)
    print("           DATASET EXPLORATION SUMMARY")
    print("=" * 60)
    print(f"\n  Shape            : {df.shape[0]} samples x {df.shape[1]} columns")  # ASCII 'x' avoids cp1252 error (Bug #15)
    print(f"  Feature columns  : {[c for c in df.columns if c != 'species']}")
    print(f"  Label column     : 'species' (for reference only)")

    print("\n── Data Types ──────────────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Missing Values ──────────────────────────────────────────")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values found ✓")
    else:
        print(missing[missing > 0].to_string())

    print("\n── Descriptive Statistics (Numerical Features) ─────────────")
    feature_cols = [c for c in df.columns if c != "species"]
    print(df[feature_cols].describe().round(3).to_string())

    print("\n── Class Distribution (Reference Only) ─────────────────────")
    print(df["species"].value_counts().to_string())
    print("=" * 60 + "\n")


def verify_data_structure(df: pd.DataFrame) -> bool:
    """
    Validate that the dataset meets K-Means requirements:
    - All feature columns are numerical.
    - No missing values after inspection.

    Returns True if all checks pass.
    """
    feature_cols = [c for c in df.columns if c != "species"]
    all_numeric = all(pd.api.types.is_numeric_dtype(df[c]) for c in feature_cols)
    no_missing = df[feature_cols].isnull().sum().sum() == 0

    print("\n── Data Validation ─────────────────────────────────────────")
    print(f"  All features numerical : {'✓' if all_numeric else '✗'}")
    print(f"  No missing values      : {'✓' if no_missing else '✗'}")

    return all_numeric and no_missing


if __name__ == "__main__":
    cfg = load_config()
    df = load_iris_data(cfg, save_raw=True)
    explore_dataset(df)
    verify_data_structure(df)
