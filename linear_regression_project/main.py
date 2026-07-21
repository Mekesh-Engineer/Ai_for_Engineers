#!/usr/bin/env python3
"""
main.py  —  Linear Regression Experiment Runner
================================================
Orchestrates the full Experiment 1 pipeline:

  1. Data loading & preprocessing
  2. Simple Linear Regression training & evaluation
  3. Multiple Linear Regression training & evaluation
  4. Cross-validation for both models
  5. Residual analysis
  6. Model comparison
  7. Visualization (all plots)
  8. Report generation
  9. Predictions CSV export
 10. Model persistence

Run from the project root:
    python main.py
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Ensure project-level imports work regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import load_config, preprocess_data
from src.model_training import (
    train_simple_regression,
    train_multiple_regression,
    predict,
    save_model,
    get_model_summary,
)
from src.evaluation import (
    evaluate_model,
    cross_validate_model,
    residual_analysis,
    compare_models,
    generate_report,
)
from src.visualization import generate_all_plots


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    print("╔" + "═" * 68 + "╗")
    print("║  EXPERIMENT 1: Simple & Multiple Linear Regression                  ║")
    print("║  California Housing Dataset — Full Pipeline                         ║")
    print("╚" + "═" * 68 + "╝")
    print(f"  Started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project    : {PROJECT_ROOT}\n")

    # ── Load Configuration ──────────────────────────────────────────────
    config = load_config(os.path.join(PROJECT_ROOT, "config", "parameters.json"))
    print("[1/10] Configuration loaded.\n")

    # ── Step 1: Preprocessing ───────────────────────────────────────────
    print("[2/10] Running data preprocessing pipeline …")
    data = preprocess_data(config)

    # ── Step 2: Simple Linear Regression ────────────────────────────────
    print("\n[3/10] Training or Loading Simple Linear Regression …")
    simple_feature = config["data"]["simple_regression_feature"]
    simple_model_path = os.path.join(PROJECT_ROOT, config["model"]["simple_lr_save_path"])
    
    if os.path.exists(simple_model_path):
        from src.model_training import load_model
        print(f"  Found existing model at {simple_model_path}, loading...")
        loaded_model = load_model(simple_model_path)
        feature_index = data["feature_names"].index(simple_feature)
        simple_result = {
            "model": loaded_model,
            "feature_name": simple_feature,
            "feature_index": feature_index,
            "coefficient": loaded_model.coef_[0],
            "intercept": loaded_model.intercept_,
        }
    else:
        simple_result = train_simple_regression(
            data["X_train_scaled"],
            data["y_train"],
            feature_name=simple_feature,
            feature_names=data["feature_names"],
        )

    # ── Step 3: Multiple Linear Regression ──────────────────────────────
    print("\n[4/10] Training or Loading Multiple Linear Regression …")
    multi_model_path = os.path.join(PROJECT_ROOT, config["model"]["multiple_lr_save_path"])
    
    if os.path.exists(multi_model_path):
        from src.model_training import load_model
        print(f"  Found existing model at {multi_model_path}, loading...")
        loaded_model = load_model(multi_model_path)
        
        coeff_df = pd.DataFrame({
            "Feature": data["feature_names"],
            "Coefficient": loaded_model.coef_,
            "Abs_Coefficient": np.abs(loaded_model.coef_),
        }).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
        
        multi_result = {
            "model": loaded_model,
            "coefficients": loaded_model.coef_,
            "intercept": loaded_model.intercept_,
            "feature_names": data["feature_names"],
            "coeff_df": coeff_df,
        }
    else:
        multi_result = train_multiple_regression(
            data["X_train_scaled"],
            data["y_train"],
            feature_names=data["feature_names"],
        )

    # ── Step 4: Evaluation ──────────────────────────────────────────────
    print("\n[5/10] Evaluating models …")
    simple_metrics = evaluate_model(
        simple_result["model"],
        data["X_train_scaled"], data["y_train"],
        data["X_test_scaled"], data["y_test"],
        model_name="Simple LR",
        feature_index=simple_result["feature_index"],
    )

    multi_metrics = evaluate_model(
        multi_result["model"],
        data["X_train_scaled"], data["y_train"],
        data["X_test_scaled"], data["y_test"],
        model_name="Multiple LR",
    )

    # ── Step 5: Cross-Validation ────────────────────────────────────────
    print("\n[6/10] Running cross-validation …")
    cv_folds = config["training"]["cross_validation_folds"]

    from sklearn.linear_model import LinearRegression as _LR

    simple_cv = cross_validate_model(
        _LR(), data["X_train_scaled"], data["y_train"],
        cv=cv_folds, scoring="r2",
        feature_index=simple_result["feature_index"],
        model_name="Simple LR",
    )
    multi_cv = cross_validate_model(
        _LR(), data["X_train_scaled"], data["y_train"],
        cv=cv_folds, scoring="r2",
        model_name="Multiple LR",
    )

    # ── Step 6: Residual Analysis ───────────────────────────────────────
    print("\n[7/10] Analysing residuals …")
    simple_residuals = residual_analysis(
        data["y_test"],
        simple_metrics["test"]["predictions"],
        model_name="Simple LR",
    )
    multi_residuals = residual_analysis(
        data["y_test"],
        multi_metrics["test"]["predictions"],
        model_name="Multiple LR",
    )

    # ── Step 7: Model Comparison ────────────────────────────────────────
    print("\n[8/10] Comparing models …")
    comparison_df = compare_models(
        simple_metrics, multi_metrics,
        simple_cv, multi_cv,
    )

    # Save comparison CSV
    comp_path = os.path.join(PROJECT_ROOT, config["output"]["comparison_file"])
    os.makedirs(os.path.dirname(comp_path), exist_ok=True)
    comparison_df.to_csv(comp_path, index=False)

    # ── Step 8: Visualization ───────────────────────────────────────────
    print("\n[9/10] Generating visualizations …")
    generate_all_plots(
        data, simple_result, multi_result,
        simple_metrics, multi_metrics,
        simple_cv, multi_cv,
        simple_residuals, multi_residuals,
        config,
    )

    # ── Step 9: Report & Predictions CSV ────────────────────────────────
    print("\n[10/10] Generating report & predictions …")
    generate_report(
        simple_metrics, multi_metrics,
        simple_cv, multi_cv,
        simple_residuals, multi_residuals,
        simple_result, multi_result,
        config,
    )

    # Predictions CSV
    predictions_df = pd.DataFrame({
        "Actual": data["y_test"].values,
        "Simple_LR_Predicted": simple_metrics["test"]["predictions"],
        "Multiple_LR_Predicted": multi_metrics["test"]["predictions"],
    })
    pred_path = os.path.join(PROJECT_ROOT, config["output"]["predictions_file"])
    predictions_df.to_csv(pred_path, index=False)
    print(f"[INFO] Predictions saved → {pred_path}")

    # ── Step 10: Save Models ────────────────────────────────────────────
    models_dir = os.path.join(PROJECT_ROOT, "models")
    save_model(
        simple_result["model"],
        os.path.join(PROJECT_ROOT, config["model"]["simple_lr_save_path"]),
        metadata={
            "type": "Simple Linear Regression",
            "feature": simple_result["feature_name"],
            "test_r2": simple_metrics["test"]["R2"],
        },
    )
    save_model(
        multi_result["model"],
        os.path.join(PROJECT_ROOT, config["model"]["multiple_lr_save_path"]),
        metadata={
            "type": "Multiple Linear Regression",
            "n_features": len(data["feature_names"]),
            "test_r2": multi_metrics["test"]["R2"],
        },
    )
    save_model(
        data["scaler"],
        os.path.join(PROJECT_ROOT, config["model"]["scaler_save_path"]),
        metadata={"type": config["preprocessing"]["scaling_method"]},
    )

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  EXPERIMENT COMPLETE                                                ║")
    print("╚" + "═" * 68 + "╝")
    print(f"  Simple LR  test R²  : {simple_metrics['test']['R2']:.6f}")
    print(f"  Multiple LR test R² : {multi_metrics['test']['R2']:.6f}")
    print(f"  Improvement         : +{multi_metrics['test']['R2'] - simple_metrics['test']['R2']:.6f}")
    print(f"\n  Outputs saved in    : {os.path.join(PROJECT_ROOT, 'results')}")
    print(f"  Models saved in     : {models_dir}")
    print(f"  Finished at         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
