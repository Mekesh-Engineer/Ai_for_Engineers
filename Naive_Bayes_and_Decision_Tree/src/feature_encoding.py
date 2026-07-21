"""
feature_encoding.py
===================
EncodingModule: Handle categorical feature encoding and train/test splitting.

Functions
---------
encode_categorical_features(df, target_col, method='LabelEncoder')
    Encode all categorical columns; return encoded DataFrame + mapping dict.
split_and_prepare_data(df, target_col, test_size, random_state)
    Separate features/target and produce stratified train/test splits.
save_processed_data(df, path)
    Persist encoded DataFrame to CSV.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def encode_categorical_features(
    df: pd.DataFrame,
    target_col: str,
    method: str = "LabelEncoder",
) -> tuple[pd.DataFrame, dict]:
    """Encode all object-dtype columns using *method*.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame (may contain the target column).
    target_col : str
        Name of the target/label column.
    method : str
        'LabelEncoder' (default) or 'OneHotEncoder'.
        OneHotEncoder is applied to feature columns only (not target).

    Returns
    -------
    df_encoded : pd.DataFrame
        DataFrame with all columns encoded as integers / dummies.
    mappings : dict
        {column_name: {original_value: encoded_int, ...}, ...}
        Useful for reversing the encoding later.
    """
    df_enc = df.copy()
    mappings: dict = {}

    categorical_cols = df_enc.select_dtypes(include=["object", "category"]).columns.tolist()

    if method == "LabelEncoder":
        for col in categorical_cols:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            mappings[col] = {str(cls): int(idx) for idx, cls in enumerate(le.classes_)}

    elif method == "OneHotEncoder":
        feature_cols = [c for c in categorical_cols if c != target_col]
        # Encode target separately with LabelEncoder
        if target_col in categorical_cols:
            le_target = LabelEncoder()
            df_enc[target_col] = le_target.fit_transform(df_enc[target_col].astype(str))
            mappings[target_col] = {
                str(cls): int(idx) for idx, cls in enumerate(le_target.classes_)
            }
        # One-hot encode feature columns
        df_enc = pd.get_dummies(df_enc, columns=feature_cols, drop_first=False)
        mappings["_ohe_columns"] = list(df_enc.columns)

    else:
        raise ValueError(f"Unknown encoding method: {method}. Use 'LabelEncoder' or 'OneHotEncoder'.")

    print("=" * 60)
    print("FEATURE ENCODING SUMMARY")
    print("=" * 60)
    print(f"  Method        : {method}")
    print(f"  Columns encoded: {list(mappings.keys())}")
    if method == "LabelEncoder":
        for col, mapping in mappings.items():
            print(f"\n  {col}:")
            for orig, enc in mapping.items():
                print(f"    {orig!r:>12} → {enc}")
    print(f"\n  Final shape   : {df_enc.shape}")
    print()

    return df_enc, mappings


def split_and_prepare_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple:
    """Separate features from target and create stratified train/test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Fully encoded DataFrame.
    target_col : str
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray or pd.DataFrame
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)
    print(f"  Total samples : {len(df)}")
    print(f"  Train samples : {len(X_train)}  ({(1 - test_size)*100:.0f}%)")
    print(f"  Test samples  : {len(X_test)}  ({test_size*100:.0f}%)")
    print(f"  Features      : {list(X.columns)}")
    print(f"  Target        : {target_col}")
    print()

    return X_train, X_test, y_train, y_test


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """Save encoded DataFrame to *path* (creates directories if needed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Processed data saved → {path}")


def save_encoding_mappings(mappings: dict, path: str) -> None:
    """Save the encoding mapping dictionary as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(mappings, f, indent=2)
    print(f"  Encoding mappings saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.data_loader import load_classification_data, check_data_quality, augment_data

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "play_tennis.csv")
    df = load_classification_data(DATA_PATH, drop_cols=["Day"])
    check_data_quality(df, "PlayTennis")
    df_aug = augment_data(df, n_samples=140)

    df_enc, mappings = encode_categorical_features(df_aug, "PlayTennis", method="LabelEncoder")
    X_train, X_test, y_train, y_test = split_and_prepare_data(df_enc, "PlayTennis")
    print("X_train shape:", X_train.shape)
    print("X_test  shape:", X_test.shape)
