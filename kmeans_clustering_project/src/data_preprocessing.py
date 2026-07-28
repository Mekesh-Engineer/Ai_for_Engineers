"""
data_preprocessing.py
----------------------
PreprocessingModule: Handles missing value imputation, outlier treatment,
and feature scaling for K-Means clustering preparation.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


def scale_features(
    df: pd.DataFrame,
    feature_cols: list,
    method: str = "StandardScaler",
    config: dict = None,
) -> tuple:
    """
    Apply feature scaling to the given columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature_cols : list
        Column names to scale.
    method : str
        'StandardScaler' (default) or 'MinMaxScaler'.
    config : dict, optional
        Project config; used to persist scaled output.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled feature matrix.
    scaler : fitted scaler object
        Fitted scaler for inverse-transforming cluster centers later.
    """
    X = df[feature_cols].values

    scaler = StandardScaler() if method == "StandardScaler" else MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n[Preprocessing] Feature scaling applied: {method}")
    print(f"  Features scaled  : {feature_cols}")
    if method == "StandardScaler":
        # Convert np.float64 -> float so they print cleanly (Bug #10)
        means = {col: float(round(v, 4)) for col, v in zip(feature_cols, scaler.mean_)}
        stds  = {col: float(round(v, 4)) for col, v in zip(feature_cols, scaler.scale_)}
        print(f"  Means            : {means}")
        print(f"  Std deviations   : {stds}")
    else:
        mins = {col: float(round(v, 4)) for col, v in zip(feature_cols, scaler.data_min_)}
        maxs = {col: float(round(v, 4)) for col, v in zip(feature_cols, scaler.data_max_)}
        print(f"  Data min         : {mins}")
        print(f"  Data max         : {maxs}")

    # Save scaled data
    if config is not None:
        processed_path = config["data"]["processed_path"]
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
        scaled_df.to_csv(processed_path, index=False)
        print(f"  Scaled data saved → {processed_path}")

    return X_scaled, scaler


def handle_missing_values(df: pd.DataFrame, feature_cols: list, strategy: str = "mean") -> pd.DataFrame:
    """
    Impute missing values using the specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list
    strategy : str
        'mean' or 'median'.

    Returns
    -------
    pd.DataFrame with imputed values.
    """
    imputer = SimpleImputer(strategy=strategy)
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    missing_count = df[feature_cols].isnull().sum().sum()
    print(f"[Preprocessing] Missing value imputation ({strategy}) applied. Remaining NaN: {missing_count}")
    return df


def detect_and_treat_outliers(df: pd.DataFrame, feature_cols: list, method: str = "IQR") -> pd.DataFrame:
    """
    Detect and treat outliers using the IQR method (cap to fence values).

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list
    method : str
        Only 'IQR' is currently supported.

    Returns
    -------
    pd.DataFrame with outliers capped.
    """
    df_clean = df.copy()
    total_outliers = 0

    print("\n[Preprocessing] Outlier detection using IQR method:")
    for col in feature_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR

        outliers_mask = (df_clean[col] < lower_fence) | (df_clean[col] > upper_fence)
        n_out = outliers_mask.sum()
        total_outliers += n_out

        # Cap outliers to fence values
        df_clean[col] = df_clean[col].clip(lower=lower_fence, upper=upper_fence)
        print(f"  {col:<25} | outliers capped: {n_out} | range [{lower_fence:.3f}, {upper_fence:.3f}]")

    print(f"  Total outliers treated: {total_outliers}")
    return df_clean


def preprocess_pipeline(df: pd.DataFrame, config: dict) -> tuple:
    """
    Full preprocessing pipeline:
      1. Handle missing values
      2. Detect and treat outliers
      3. Scale features

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset (including 'species' reference column).
    config : dict
        Project configuration.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled feature matrix ready for K-Means.
    feature_cols : list
        Names of the scaled features.
    scaler : fitted scaler object
    df_clean : pd.DataFrame
        Cleaned dataframe before scaling.
    """
    feature_cols = [c for c in df.columns if c != "species"]
    df_clean = df.copy()

    # Step 1 – Missing value imputation
    missing_strategy = config["preprocessing"].get("handle_missing", "mean").replace("_imputation", "")
    df_clean = handle_missing_values(df_clean, feature_cols, strategy=missing_strategy)

    # Step 2 – Outlier treatment
    outlier_method = config["preprocessing"].get("outlier_method", "IQR")
    df_clean = detect_and_treat_outliers(df_clean, feature_cols, method=outlier_method)

    # Step 3 – Feature scaling
    scaler_method = config["preprocessing"].get("scaler", "StandardScaler")
    X_scaled, scaler = scale_features(df_clean, feature_cols, method=scaler_method, config=config)

    return X_scaled, feature_cols, scaler, df_clean


if __name__ == "__main__":
    from data_loader import load_config, load_iris_data, explore_dataset

    cfg = load_config()
    df  = load_iris_data(cfg, save_raw=True)
    explore_dataset(df)

    X_scaled, feature_cols, scaler, df_clean = preprocess_pipeline(df, cfg)
    print(f"\n[Preprocessing] Final scaled matrix shape: {X_scaled.shape}")
    print(f"[Preprocessing] Features: {feature_cols}")
