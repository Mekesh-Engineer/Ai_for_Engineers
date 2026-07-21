"""
Data Preprocessing Module
=========================
Handles data loading, cleaning, missing value imputation, outlier treatment,
feature scaling, and train-test splitting for the Linear Regression project.

Functions:
    load_housing_data()      – Fetch California Housing and save raw CSV
    explore_data()           – Summary statistics and missing-value report
    handle_missing_values()  – Impute missing values (mean / median / ffill)
    remove_outliers()        – IQR or Z-score based outlier removal
    scale_features()         – StandardScaler or MinMaxScaler normalization
    split_data()             – Train-test split with reproducibility
    preprocess_data()        – Full pipeline orchestrator
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = None) -> dict:
    """Load parameters from the JSON configuration file."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "parameters.json",
        )
    with open(config_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_housing_data(config: dict = None) -> pd.DataFrame:
    """
    Load the California Housing dataset from scikit-learn, convert it to a
    DataFrame, and persist the raw CSV to ``data/raw/``.

    Parameters
    ----------
    config : dict, optional
        Project configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Raw housing data with feature columns and target ``MedHouseVal``.
    """
    if config is None:
        config = load_config()

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # already contains target column 'MedHouseVal'

    # Persist raw data
    raw_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["data"]["raw_data_path"],
    )
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df.to_csv(raw_path, index=False)

    print(f"[INFO] Loaded California Housing dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[INFO] Raw data saved to {raw_path}")
    return df


# ---------------------------------------------------------------------------
# Exploratory Summary
# ---------------------------------------------------------------------------

def explore_data(df: pd.DataFrame) -> dict:
    """
    Print summary statistics and return a diagnostic dictionary.

    Returns
    -------
    dict
        Keys: 'shape', 'dtypes', 'missing', 'describe'.
    """
    print("\n" + "=" * 60)
    print("DATA EXPLORATION SUMMARY")
    print("=" * 60)
    print(f"\nShape         : {df.shape}")
    print(f"Data types    :\n{df.dtypes}\n")

    missing = df.isnull().sum()
    print(f"Missing values:\n{missing[missing > 0] if missing.sum() > 0 else 'None'}\n")

    print("Descriptive statistics:")
    print(df.describe().round(3).to_string())

    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing": missing.to_dict(),
        "describe": df.describe().to_dict(),
    }


# ---------------------------------------------------------------------------
# Missing Value Handling
# ---------------------------------------------------------------------------

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
) -> pd.DataFrame:
    """
    Impute missing values in numerical columns.

    Parameters
    ----------
    strategy : {'mean', 'median', 'ffill'}
        Imputation strategy.

    Returns
    -------
    pd.DataFrame
        DataFrame with no missing values.
    """
    num_missing_before = df.isnull().sum().sum()

    if num_missing_before == 0:
        print("[INFO] No missing values detected — skipping imputation.")
        return df.copy()

    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    if strategy == "mean":
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
            df_clean[numeric_cols].mean()
        )
    elif strategy == "median":
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
            df_clean[numeric_cols].median()
        )
    elif strategy == "ffill":
        df_clean[numeric_cols] = df_clean[numeric_cols].ffill()
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'mean', 'median', or 'ffill'.")

    print(f"[INFO] Imputed {num_missing_before} missing values using '{strategy}' strategy.")
    return df_clean


# ---------------------------------------------------------------------------
# Outlier Treatment
# ---------------------------------------------------------------------------

def remove_outliers(
    df: pd.DataFrame,
    method: str = "IQR",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    target_col: str = "MedHouseVal",
) -> pd.DataFrame:
    """
    Remove outliers from numerical columns using IQR or Z-score method.

    Parameters
    ----------
    method : {'IQR', 'zscore'}
    iqr_multiplier : float
        Multiplier for IQR method (default 1.5).
    z_threshold : float
        Threshold for Z-score method (default 3.0).
    target_col : str
        Name of target column (also treated for outliers).

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers removed.
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    rows_before = len(df_clean)

    if method.upper() == "IQR":
        for col in numeric_cols:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    elif method.lower() == "zscore":
        z_scores = np.abs(stats.zscore(df_clean[numeric_cols]))
        df_clean = df_clean[(z_scores < z_threshold).all(axis=1)]

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'IQR' or 'zscore'.")

    rows_removed = rows_before - len(df_clean)
    print(f"[INFO] Outlier removal ({method}): {rows_removed} rows removed "
          f"({rows_removed / rows_before * 100:.1f}%)  —  {len(df_clean)} rows remain.")
    return df_clean.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------------------------

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "StandardScaler",
):
    """
    Fit a scaler on training features and transform both train and test sets.

    Parameters
    ----------
    method : {'StandardScaler', 'MinMaxScaler'}

    Returns
    -------
    (np.ndarray, np.ndarray, scaler object)
    """
    if method == "StandardScaler":
        scaler = StandardScaler()
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler '{method}'.")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[INFO] Features scaled using {method}.")
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Train-Test Split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target_col: str = "MedHouseVal",
    test_size: float = 0.2,
    random_seed: int = 42,
):
    """
    Separate features and target, then split into train and test sets.

    Returns
    -------
    (X_train, X_test, y_train, y_test) – all as DataFrames / Series
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed,
    )

    print(f"[INFO] Data split: train={len(X_train)}, test={len(X_test)} "
          f"(ratio {1 - test_size:.0%} / {test_size:.0%})")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Correlation Analysis
# ---------------------------------------------------------------------------

def analyze_correlations(
    df: pd.DataFrame,
    target_col: str = "MedHouseVal",
    threshold: float = 0.90,
) -> tuple:
    """
    Compute correlation matrix, identify highly correlated feature pairs,
    and rank features by correlation with the target.

    Returns
    -------
    (correlation_matrix, high_corr_pairs, feature_target_corr)
    """
    corr_matrix = df.corr()
    feature_target = corr_matrix[target_col].drop(target_col).sort_values(
        ascending=False, key=abs
    )

    # Identify highly correlated feature pairs (excl. target)
    features_only = corr_matrix.drop(columns=[target_col], index=[target_col])
    high_corr = []
    cols = features_only.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = abs(features_only.iloc[i, j])
            if r >= threshold:
                high_corr.append((cols[i], cols[j], round(r, 4)))

    print(f"\n[INFO] Feature-target correlations (|r|):")
    for feat, val in feature_target.items():
        print(f"       {feat:15s} → {val:+.4f}")

    if high_corr:
        print(f"\n[WARN] Highly correlated pairs (|r| ≥ {threshold}):")
        for a, b, r in high_corr:
            print(f"       {a} ↔ {b} : {r}")
    else:
        print(f"\n[INFO] No feature pairs exceed correlation threshold {threshold}.")

    return corr_matrix, high_corr, feature_target


# ---------------------------------------------------------------------------
# Full Preprocessing Pipeline
# ---------------------------------------------------------------------------

def preprocess_data(config: dict = None):
    """
    Execute the complete preprocessing pipeline:
      1. Load data
      2. Explore
      3. Handle missing values
      4. Remove outliers
      5. Correlation analysis
      6. Split data
      7. Scale features

    Returns
    -------
    dict with keys:
        'X_train', 'X_test', 'y_train', 'y_test',
        'X_train_scaled', 'X_test_scaled', 'scaler',
        'feature_names', 'corr_matrix', 'df_clean', 'exploration'
    """
    if config is None:
        config = load_config()

    # Step 1 – Load
    df = load_housing_data(config)

    # Step 2 – Explore
    exploration = explore_data(df)

    # Step 3 – Missing values
    df = handle_missing_values(
        df, strategy=config["preprocessing"]["missing_value_strategy"]
    )

    # Step 4 – Outliers
    df = remove_outliers(
        df,
        method=config["preprocessing"]["outlier_method"],
        iqr_multiplier=config["preprocessing"]["outlier_iqr_multiplier"],
    )

    # Step 5 – Correlations
    corr_matrix, high_corr_pairs, feat_target_corr = analyze_correlations(
        df,
        target_col=config["data"]["target_column"],
        threshold=config["preprocessing"]["correlation_threshold"],
    )

    # Save cleaned data
    processed_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["data"]["processed_data_path"],
    )
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"[INFO] Cleaned data saved to {processed_path}")

    # Step 6 – Split
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col=config["data"]["target_column"],
        test_size=config["training"]["test_size"],
        random_seed=config["training"]["random_seed"],
    )

    # Step 7 – Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test,
        method=config["preprocessing"]["scaling_method"],
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_names": list(X_train.columns),
        "corr_matrix": corr_matrix,
        "df_clean": df,
        "exploration": exploration,
    }


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = preprocess_data()
    print(f"\n✔ Preprocessing complete.  "
          f"Training samples: {len(result['y_train'])}, "
          f"Test samples: {len(result['y_test'])}")
