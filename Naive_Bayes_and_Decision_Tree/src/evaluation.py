"""
evaluation.py
=============
EvaluationModule: Compute classification metrics, cross-validation, and reports.

Functions
---------
evaluate_classification(model, X_test, y_test, model_name, class_names)
    Full metric suite: accuracy, precision, recall, F1, confusion matrix.
cross_validate_model(model, X, y, cv, model_name)
    k-fold cross-validation with mean ± std reporting.
generate_roc_data(model, X_test, y_test)
    Produce FPR / TPR / AUC for ROC curve plotting.
compare_models(results)
    Build and save a comparative DataFrame + text report.
extract_tree_rules(model, feature_names, class_names)
    Export human-readable decision rules from a Decision Tree.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import export_text


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_classification(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
    class_names: list = None,
) -> dict:
    """Compute and print a full classification metric suite.

    Parameters
    ----------
    model : fitted sklearn classifier
    X_test : array-like
    y_test : array-like
    model_name : str
    class_names : list, optional
        Human-readable class labels for the report.

    Returns
    -------
    dict
        Keys: model_name, accuracy, precision, recall, f1, confusion_matrix,
              predictions, probabilities (if model supports predict_proba).
    """
    y_pred = model.predict(X_test)

    # Probability scores (for ROC-AUC)
    has_proba = hasattr(model, "predict_proba")
    y_prob = model.predict_proba(X_test)[:, 1] if has_proba else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    # Specificity (TN / (TN + FP)) for binary case
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = None

    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    print("=" * 60)
    print(f"EVALUATION — {model_name.upper()}")
    print("=" * 60)
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {prec:.4f}  (weighted)")
    print(f"  Recall     : {rec:.4f}  (weighted)")
    print(f"  F1-Score   : {f1:.4f}  (weighted)")
    if specificity is not None:
        print(f"  Specificity: {specificity:.4f}")
    if auc is not None:
        print(f"  ROC-AUC    : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(cm)

    cr = classification_report(
        y_test, y_pred,
        target_names=class_names if class_names else None,
        zero_division=0,
    )
    print(f"\n  Classification Report:\n{cr}")

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": specificity,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "probabilities": y_prob,
        "classification_report": cr,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────────────────────────────────────

def cross_validate_model(
    model,
    X,
    y,
    cv: int = 5,
    model_name: str = "Model",
) -> dict:
    """Stratified k-fold cross-validation.

    Parameters
    ----------
    model : sklearn estimator (unfitted clone used internally)
    X, y  : full feature matrix and label vector
    cv    : number of folds
    model_name : str

    Returns
    -------
    dict  {model_name, cv_scores, mean, std}
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

    print("=" * 60)
    print(f"CROSS-VALIDATION — {model_name.upper()}  ({cv}-fold Stratified)")
    print("=" * 60)
    print(f"  Fold scores : {np.round(scores, 4)}")
    print(f"  Mean        : {scores.mean():.4f}")
    print(f"  Std         : {scores.std():.4f}")
    print()

    return {
        "model_name": model_name,
        "cv_scores": scores,
        "mean": scores.mean(),
        "std": scores.std(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# ROC data
# ──────────────────────────────────────────────────────────────────────────────

def generate_roc_data(model, X_test, y_test) -> dict:
    """Compute ROC curve data points.

    Returns
    -------
    dict  {fpr, tpr, thresholds, auc}  or None if model lacks predict_proba.
    """
    if not hasattr(model, "predict_proba"):
        print("  Model does not support predict_proba; ROC skipped.")
        return None

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}


# ──────────────────────────────────────────────────────────────────────────────
# Model comparison report
# ──────────────────────────────────────────────────────────────────────────────

def compare_models(results: list[dict], save_dir: str = "results") -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame and save reports.

    Parameters
    ----------
    results : list of dicts returned by evaluate_classification()
    save_dir : str

    Returns
    -------
    pd.DataFrame  Comparison table.
    """
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for r in results:
        rows.append({
            "Model":       r["model_name"],
            "Accuracy":    round(r["accuracy"],    4),
            "Precision":   round(r["precision"],   4),
            "Recall":      round(r["recall"],      4),
            "F1-Score":    round(r["f1_score"],    4),
            "Specificity": round(r["specificity"], 4) if r["specificity"] is not None else "N/A",
            "ROC-AUC":     round(r["roc_auc"],     4) if r["roc_auc"] is not None else "N/A",
        })

    df_cmp = pd.DataFrame(rows).set_index("Model")

    # Save CSV
    csv_path = os.path.join(save_dir, "performance_comparison.csv")
    df_cmp.to_csv(csv_path)
    print(f"\n  Performance comparison saved → {csv_path}")

    # Save text report
    txt_path = os.path.join(save_dir, "classification_report.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("CLASSIFICATION PERFORMANCE COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        f.write(df_cmp.to_string())
        f.write("\n\n")
        for r in results:
            f.write("=" * 60 + "\n")
            f.write(f"{r['model_name']} — Full Classification Report\n")
            f.write("=" * 60 + "\n")
            f.write(r["classification_report"])
            f.write("\n")

    print(f"  Classification report saved → {txt_path}")

    print("\n  ─── PERFORMANCE COMPARISON ───")
    print(df_cmp.to_string())
    print()

    return df_cmp


# ──────────────────────────────────────────────────────────────────────────────
# Decision Tree rule extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_tree_rules(
    model,
    feature_names: list,
    class_names: list = None,
    save_path: str = None,
) -> str:
    """Export human-readable decision rules from a fitted DecisionTreeClassifier.

    Parameters
    ----------
    model : fitted DecisionTreeClassifier
    feature_names : list of str
    class_names : list of str, optional
    save_path : str, optional  Path to save the rules as a .txt file.

    Returns
    -------
    str  Text representation of the decision tree rules.
    """
    rules = export_text(
        model,
        feature_names=feature_names,
        class_names=class_names,
        show_weights=True,
    )

    print("=" * 60)
    print("DECISION TREE RULES")
    print("=" * 60)
    print(rules)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(rules)
        print(f"  Decision rules saved → {save_path}")

    return rules


# ──────────────────────────────────────────────────────────────────────────────
# Misclassification analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_misclassifications(X_test, y_test, y_pred, feature_names: list) -> pd.DataFrame:
    """Return a DataFrame of misclassified instances for manual inspection."""
    import pandas as pd
    df = pd.DataFrame(X_test, columns=feature_names)
    df["true_label"]      = np.array(y_test)
    df["predicted_label"] = y_pred
    df["correct"]         = df["true_label"] == df["predicted_label"]
    misclf = df[~df["correct"]].reset_index(drop=True)
    print(f"  Misclassified instances: {len(misclf)} / {len(df)}")
    return misclf
