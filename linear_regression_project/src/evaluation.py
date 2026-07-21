"""
Evaluation Module
=================
Calculate regression performance metrics, perform cross-validation,
analyse residuals, and generate comprehensive comparison reports.

Functions:
    evaluate_model()          – R², MSE, RMSE, MAE for train & test
    cross_validate_model()    – K-fold cross-validation scores
    residual_analysis()       – Residual statistics & normality test
    compare_models()          – Side-by-side simple vs. multiple
    generate_report()         – Write performance_metrics.txt
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_train, y_train,
    X_test, y_test,
    model_name: str = "Linear Regression",
    feature_index: int = None,
) -> dict:
    """
    Compute R², MSE, RMSE, MAE on both training and test sets.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_train, X_test : array-like
    y_train, y_test : array-like
    model_name : str
    feature_index : int, optional
        If given, extract a single column (for simple regression).

    Returns
    -------
    dict  with 'train' and 'test' sub-dicts of metrics.
    """
    def _prepare(X):
        X = np.array(X)
        if feature_index is not None:
            X = X[:, feature_index].reshape(-1, 1)
        return X

    X_tr = _prepare(X_train)
    X_te = _prepare(X_test)
    y_tr = np.array(y_train)
    y_te = np.array(y_test)

    train_pred = model.predict(X_tr)
    test_pred = model.predict(X_te)

    metrics = {}
    for split_name, y_true, y_pred in [
        ("train", y_tr, train_pred),
        ("test", y_te, test_pred),
    ]:
        mse = mean_squared_error(y_true, y_pred)
        metrics[split_name] = {
            "R2": r2_score(y_true, y_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_true, y_pred),
            "predictions": y_pred,
        }

    print(f"\n{'=' * 60}")
    print(f"EVALUATION — {model_name}")
    print(f"{'=' * 60}")
    for split in ("train", "test"):
        m = metrics[split]
        print(f"\n  [{split.upper()}]")
        print(f"    R² Score : {m['R2']:.6f}")
        print(f"    MSE      : {m['MSE']:.6f}")
        print(f"    RMSE     : {m['RMSE']:.6f}")
        print(f"    MAE      : {m['MAE']:.6f}")

    return metrics


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    X, y,
    cv: int = 5,
    scoring: str = "r2",
    feature_index: int = None,
    model_name: str = "Linear Regression",
) -> dict:
    """
    Perform k-fold cross-validation and return per-fold and aggregate scores.

    Returns
    -------
    dict  with 'scores', 'mean', 'std'.
    """
    X = np.array(X)
    if feature_index is not None:
        X = X[:, feature_index].reshape(-1, 1)

    scores = cross_val_score(model, X, np.array(y), cv=cv, scoring=scoring)

    print(f"\n{'=' * 60}")
    print(f"CROSS-VALIDATION — {model_name}  ({cv}-fold, scoring={scoring})")
    print(f"{'=' * 60}")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.6f}")
    print(f"\n  Mean  : {scores.mean():.6f}")
    print(f"  Std   : {scores.std():.6f}")

    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}


# ---------------------------------------------------------------------------
# Residual Analysis
# ---------------------------------------------------------------------------

def residual_analysis(y_true, y_pred, model_name: str = "Model") -> dict:
    """
    Analyse prediction residuals: basic statistics and Shapiro-Wilk
    normality test.

    Returns
    -------
    dict  with 'residuals', 'mean', 'std', 'skew', 'kurtosis',
          'shapiro_stat', 'shapiro_p'.
    """
    residuals = np.array(y_true) - np.array(y_pred)

    # Shapiro-Wilk (limited to 5000 samples for performance)
    sample = residuals if len(residuals) <= 5000 else np.random.choice(residuals, 5000, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(sample)

    result = {
        "residuals": residuals,
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "skew": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
    }

    print(f"\n{'=' * 60}")
    print(f"RESIDUAL ANALYSIS — {model_name}")
    print(f"{'=' * 60}")
    print(f"  Mean residual : {result['mean']:.6f}")
    print(f"  Std residual  : {result['std']:.6f}")
    print(f"  Skewness      : {result['skew']:.4f}")
    print(f"  Kurtosis      : {result['kurtosis']:.4f}")
    print(f"  Shapiro-Wilk  : W={result['shapiro_stat']:.4f}, p={result['shapiro_p']:.4e}")
    if result['shapiro_p'] < 0.05:
        print("  ⚠  Residuals are NOT normally distributed (p < 0.05)")
    else:
        print("  ✔  Residuals appear normally distributed (p ≥ 0.05)")

    return result


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

def compare_models(
    simple_metrics: dict,
    multi_metrics: dict,
    simple_cv: dict = None,
    multi_cv: dict = None,
) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame of Simple vs. Multiple LR.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for metric_name in ("R2", "MSE", "RMSE", "MAE"):
        rows.append({
            "Metric": metric_name,
            "Simple_Train": simple_metrics["train"][metric_name],
            "Simple_Test": simple_metrics["test"][metric_name],
            "Multiple_Train": multi_metrics["train"][metric_name],
            "Multiple_Test": multi_metrics["test"][metric_name],
        })

    if simple_cv and multi_cv:
        rows.append({
            "Metric": "CV_R2_Mean",
            "Simple_Train": simple_cv["mean"],
            "Simple_Test": simple_cv["mean"],
            "Multiple_Train": multi_cv["mean"],
            "Multiple_Test": multi_cv["mean"],
        })
        rows.append({
            "Metric": "CV_R2_Std",
            "Simple_Train": simple_cv["std"],
            "Simple_Test": simple_cv["std"],
            "Multiple_Train": multi_cv["std"],
            "Multiple_Test": multi_cv["std"],
        })

    comparison_df = pd.DataFrame(rows)

    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON — Simple vs. Multiple Linear Regression")
    print(f"{'=' * 80}")
    print(comparison_df.to_string(index=False, float_format="{:.6f}".format))

    return comparison_df


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(
    simple_metrics: dict,
    multi_metrics: dict,
    simple_cv: dict,
    multi_cv: dict,
    simple_residuals: dict,
    multi_residuals: dict,
    simple_result: dict,
    multi_result: dict,
    config: dict,
    output_path: str = None,
) -> str:
    """
    Write a comprehensive performance report to a text file.
    """
    if output_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base, config["output"]["metrics_file"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("LINEAR REGRESSION — PERFORMANCE REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Configuration summary
    lines.append("\n── Configuration ──")
    lines.append(f"  Dataset          : {config['data']['dataset_name']}")
    lines.append(f"  Test size        : {config['training']['test_size']}")
    lines.append(f"  Scaling          : {config['preprocessing']['scaling_method']}")
    lines.append(f"  CV folds         : {config['training']['cross_validation_folds']}")
    lines.append(f"  Random seed      : {config['training']['random_seed']}")
    lines.append(f"  Outlier method   : {config['preprocessing']['outlier_method']}")

    # Simple LR
    lines.append("\n── Simple Linear Regression ──")
    lines.append(f"  Feature : {simple_result.get('feature_name', 'N/A')}")
    lines.append(f"  Slope   : {simple_result.get('coefficient', 0):.6f}")
    lines.append(f"  Intercept: {simple_result.get('intercept', 0):.6f}")
    for split in ("train", "test"):
        m = simple_metrics[split]
        lines.append(f"\n  [{split.upper()}]")
        for k in ("R2", "MSE", "RMSE", "MAE"):
            lines.append(f"    {k:6s}: {m[k]:.6f}")
    lines.append(f"\n  Cross-Validation R² : {simple_cv['mean']:.6f} ± {simple_cv['std']:.6f}")
    lines.append(f"  Residual skewness   : {simple_residuals['skew']:.4f}")
    lines.append(f"  Residual kurtosis   : {simple_residuals['kurtosis']:.4f}")

    # Multiple LR
    lines.append("\n── Multiple Linear Regression ──")
    lines.append(f"  Intercept: {multi_result.get('intercept', 0):.6f}")
    if "coeff_df" in multi_result:
        lines.append("  Coefficients:")
        for _, row in multi_result["coeff_df"].iterrows():
            lines.append(f"    {row['Feature']:15s}: {row['Coefficient']:+.6f}")
    for split in ("train", "test"):
        m = multi_metrics[split]
        lines.append(f"\n  [{split.upper()}]")
        for k in ("R2", "MSE", "RMSE", "MAE"):
            lines.append(f"    {k:6s}: {m[k]:.6f}")
    lines.append(f"\n  Cross-Validation R² : {multi_cv['mean']:.6f} ± {multi_cv['std']:.6f}")
    lines.append(f"  Residual skewness   : {multi_residuals['skew']:.4f}")
    lines.append(f"  Residual kurtosis   : {multi_residuals['kurtosis']:.4f}")

    # Comparison
    lines.append("\n── Summary Comparison ──")
    r2_simple = simple_metrics["test"]["R2"]
    r2_multi = multi_metrics["test"]["R2"]
    improvement = r2_multi - r2_simple
    lines.append(f"  Simple LR test R²   : {r2_simple:.6f}")
    lines.append(f"  Multiple LR test R² : {r2_multi:.6f}")
    lines.append(f"  R² improvement      : +{improvement:.6f} ({improvement / max(abs(r2_simple), 1e-9) * 100:.1f}%)")

    if r2_multi > r2_simple:
        lines.append("  → Multiple LR outperforms Simple LR as expected.")
    else:
        lines.append("  → Simple LR performed comparably or better (unusual).")

    lines.append("\n" + "=" * 70)

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[INFO] Performance report saved → {output_path}")
    return report_text
