"""
Model Training Module
=====================
Train Simple (univariate) and Multiple (multivariate) Linear Regression
models, persist them to disk, and expose convenience functions for
prediction and coefficient inspection.

Functions:
    train_simple_regression()    – Fit y = mx + b on a single feature
    train_multiple_regression()  – Fit y = b₀ + b₁x₁ + … + bₙxₙ
    save_model()                 – Serialize model to .pkl
    load_model()                 – Deserialize model from .pkl
    get_model_summary()          – Pretty-print coefficients & intercept
    predict()                    – Generate predictions from a trained model
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Helper – config loader (mirrored for standalone use)
# ---------------------------------------------------------------------------

def _load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "parameters.json",
        )
    with open(config_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Simple Linear Regression
# ---------------------------------------------------------------------------

def train_simple_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_name: str = "MedInc",
    feature_index: int = None,
    feature_names: list = None,
) -> dict:
    """
    Train a Simple Linear Regression model  y = m·x + b  using a single
    feature column.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Full training feature matrix (scaled or unscaled).
    y_train : array-like, shape (n_samples,)
        Training target values.
    feature_name : str
        Name of the feature to use (for logging).
    feature_index : int, optional
        Column index of the selected feature.  If None, it is looked up
        from *feature_names*.
    feature_names : list, optional
        Ordered list of feature names corresponding to columns in X_train.

    Returns
    -------
    dict
        'model', 'feature_name', 'feature_index', 'coefficient', 'intercept',
        'train_predictions'.
    """
    # Resolve feature index
    if feature_index is None:
        if feature_names is not None and feature_name in feature_names:
            feature_index = feature_names.index(feature_name)
        else:
            feature_index = 0  # fallback to first column

    # Extract single column
    if isinstance(X_train, pd.DataFrame):
        X_single = X_train.iloc[:, feature_index].values.reshape(-1, 1)
    else:
        X_single = np.array(X_train)[:, feature_index].reshape(-1, 1)

    y = np.array(y_train)

    model = LinearRegression()
    model.fit(X_single, y)
    train_preds = model.predict(X_single)

    print("\n" + "=" * 60)
    print("SIMPLE LINEAR REGRESSION — TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Feature          : {feature_name} (col {feature_index})")
    print(f"  Coefficient (m)  : {model.coef_[0]:.6f}")
    print(f"  Intercept   (b)  : {model.intercept_:.6f}")
    print(f"  Equation         : y = {model.coef_[0]:.4f}·x + {model.intercept_:.4f}")

    return {
        "model": model,
        "feature_name": feature_name,
        "feature_index": feature_index,
        "coefficient": model.coef_[0],
        "intercept": model.intercept_,
        "train_predictions": train_preds,
    }


# ---------------------------------------------------------------------------
# Multiple Linear Regression
# ---------------------------------------------------------------------------

def train_multiple_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list = None,
) -> dict:
    """
    Train a Multiple Linear Regression model  y = b₀ + Σ bᵢ·xᵢ  using all
    provided features.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
    y_train : array-like
    feature_names : list, optional

    Returns
    -------
    dict
        'model', 'coefficients', 'intercept', 'feature_names',
        'train_predictions', 'coeff_df'.
    """
    X = np.array(X_train)
    y = np.array(y_train)

    model = LinearRegression()
    model.fit(X, y)
    train_preds = model.predict(X)

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    coeff_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_,
        "Abs_Coefficient": np.abs(model.coef_),
    }).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("MULTIPLE LINEAR REGRESSION — TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Number of features : {X.shape[1]}")
    print(f"  Intercept (b₀)    : {model.intercept_:.6f}")
    print(f"\n  Coefficients (sorted by |value|):")
    for _, row in coeff_df.iterrows():
        print(f"    {row['Feature']:15s} → {row['Coefficient']:+.6f}")

    return {
        "model": model,
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "feature_names": feature_names,
        "train_predictions": train_preds,
        "coeff_df": coeff_df,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model, X, feature_index: int = None) -> np.ndarray:
    """
    Generate predictions.  For simple regression, extract the appropriate
    column first.
    """
    X = np.array(X)
    if feature_index is not None:
        X = X[:, feature_index].reshape(-1, 1)
    return model.predict(X)


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str, metadata: dict = None) -> str:
    """Save a trained model (and optional metadata) via joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "model": model,
        "saved_at": datetime.now().isoformat(),
        "metadata": metadata or {},
    }
    joblib.dump(payload, path)
    print(f"[INFO] Model saved → {path}")
    return path


def load_model(path: str):
    """Load a previously saved model."""
    payload = joblib.load(path)
    
    if isinstance(payload, dict) and "model" in payload:
        print(f"[INFO] Model loaded ← {path}  (saved at {payload.get('saved_at', '?')})")
        return payload["model"]
    else:
        # Handling the case where the model was saved directly without our payload wrapper
        print(f"[INFO] Model loaded ← {path}  (direct object)")
        return payload


# ---------------------------------------------------------------------------
# Summary Helper
# ---------------------------------------------------------------------------

def get_model_summary(result: dict) -> str:
    """Return a formatted string summarizing model coefficients."""
    lines = []
    if "coefficient" in result:  # simple
        lines.append("Simple Linear Regression")
        lines.append(f"  Feature   : {result['feature_name']}")
        lines.append(f"  Slope (m) : {result['coefficient']:.6f}")
        lines.append(f"  Intercept : {result['intercept']:.6f}")
    else:  # multiple
        lines.append("Multiple Linear Regression")
        lines.append(f"  Intercept : {result['intercept']:.6f}")
        for name, coef in zip(result["feature_names"], result["coefficients"]):
            lines.append(f"  {name:15s} : {coef:+.6f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_preprocessing import preprocess_data

    config = _load_config()
    data = preprocess_data(config)

    # Simple
    simple_result = train_simple_regression(
        data["X_train_scaled"], data["y_train"],
        feature_name=config["data"]["simple_regression_feature"],
        feature_names=data["feature_names"],
    )

    # Multiple
    multi_result = train_multiple_regression(
        data["X_train_scaled"], data["y_train"],
        feature_names=data["feature_names"],
    )

    print("\n✔ Both models trained successfully.")
