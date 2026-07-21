"""
live_comparative_test.py
========================
Live comparative testing tool for Naïve Bayes and Decision Tree models.
Loads the dataset, trains/tunes both models live, evaluates accuracy,
precision, recall, F1, specificity, ROC-AUC, and displays side-by-side
confusion matrices and scenario predictions.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_classification_data, check_data_quality
from src.feature_encoding import encode_categorical_features, split_and_prepare_data
from src.model_training import tune_naive_bayes, tune_decision_tree
from src.evaluation import evaluate_classification

# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions for Visual Text Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────

def print_confusion_matrix_box(cm: np.ndarray, model_name: str, class_names: list):
    """Print an ASCII formatted confusion matrix."""
    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ CONFUSION MATRIX — {model_name:<36} │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │                   Predicted: {class_names[0]:<6}   Predicted: {class_names[1]:<6} │")
    print(f"  │ Actual: {class_names[0]:<7}   {cm[0][0]:^15}   {cm[0][1]:^15} │")
    print(f"  │ Actual: {class_names[1]:<7}   {cm[1][0]:^15}   {cm[1][1]:^15} │")
    print(f"  └─────────────────────────────────────────────────────────┘")


# ──────────────────────────────────────────────────────────────────────────────
# Main Live Execution Function
# ──────────────────────────────────────────────────────────────────────────────

def run_live_comparative_suite():
    """Run full live comparative training, evaluation, and scenario testing."""
    print("=" * 80)
    print("      LIVE COMPARATIVE EXPERIMENT — NAÏVE BAYES vs DECISION TREE")
    print("=" * 80)

    # 1. Load Dataset
    raw_path = os.path.join(PROJECT_ROOT, "data", "raw", "play_tennis.csv")
    df_raw = load_classification_data(raw_path, drop_cols=["Day", "Date"])
    check_data_quality(df_raw, "PlayTennis")

    # 2. Encode Features
    df_enc, mappings = encode_categorical_features(df_raw, "PlayTennis", method="LabelEncoder")

    # 3. Train / Test Split (70/30 Stratified)
    X_train, X_test, y_train, y_test = split_and_prepare_data(
        df_enc, "PlayTennis", test_size=0.3, random_state=42
    )
    CLASS_NAMES = [k for k, v in sorted(mappings["PlayTennis"].items(), key=lambda x: x[1])]

    # 4. Train & Tune Naïve Bayes
    print("\n⏳ Fine-tuning Gaussian Naïve Bayes...")
    t0 = time.perf_counter()
    nb_model, nb_best_params = tune_naive_bayes(X_train, y_train, cv=5)
    nb_train_time = (time.perf_counter() - t0) * 1000  # in ms

    # 5. Train & Tune Decision Tree
    print("⏳ Fine-tuning Decision Tree Classifier...")
    t0 = time.perf_counter()
    dt_model, dt_best_params = tune_decision_tree(X_train, y_train, cv=5, random_state=42)
    dt_train_time = (time.perf_counter() - t0) * 1000  # in ms

    # 6. Evaluate Models
    print("\n" + "=" * 80)
    print("                     MODEL EVALUATION & METRICS SUMMARY")
    print("=" * 80)

    nb_eval = evaluate_classification(nb_model, X_test, y_test, "Gaussian Naïve Bayes (Tuned)", CLASS_NAMES)
    dt_eval = evaluate_classification(dt_model, X_test, y_test, "Decision Tree (Tuned)", CLASS_NAMES)

    # 7. Side-by-Side Performance Comparison Table
    print("\n" + "=" * 80)
    print("                  LIVE PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    print(f"  {'Metric':<20} │ {'Gaussian Naïve Bayes':<25} │ {'Decision Tree':<25}")
    print("  " + "─" * 76)
    print(f"  {'Accuracy':<20} │ {nb_eval['accuracy']*100:>22.2f}% │ {dt_eval['accuracy']*100:>22.2f}%")
    print(f"  {'Precision (Weighted)':<20} │ {nb_eval['precision']*100:>22.2f}% │ {dt_eval['precision']*100:>22.2f}%")
    print(f"  {'Recall (Weighted)':<20} │ {nb_eval['recall']*100:>22.2f}% │ {dt_eval['recall']*100:>22.2f}%")
    print(f"  {'F1-Score (Weighted)':<20} │ {nb_eval['f1_score']*100:>22.2f}% │ {dt_eval['f1_score']*100:>22.2f}%")
    print(f"  {'Specificity':<20} │ {nb_eval['specificity']*100:>22.2f}% │ {dt_eval['specificity']*100:>22.2f}%")
    print(f"  {'ROC-AUC Score':<20} │ {nb_eval['roc_auc']*100:>22.2f}% │ {dt_eval['roc_auc']*100:>22.2f}%")
    print(f"  {'Training Time':<20} │ {nb_train_time:>22.2f} ms │ {dt_train_time:>22.2f} ms")
    print("=" * 80)

    # 8. Confusion Matrices
    print("\n" + "=" * 80)
    print("                        CONFUSION MATRICES")
    print("=" * 80)
    print_confusion_matrix_box(nb_eval["confusion_matrix"], "Gaussian Naïve Bayes (Tuned)", CLASS_NAMES)
    print_confusion_matrix_box(dt_eval["confusion_matrix"], "Decision Tree (Tuned)", CLASS_NAMES)

    # 9. Sample Test Scenarios Inference
    print("\n" + "=" * 80)
    print("                     REAL-TIME INFERENCE SCENARIO TESTS")
    print("=" * 80)

    MONTH_MAP = mappings["Month"]
    OUTLOOK_MAP = mappings["Outlook"]
    TEMP_MAP = mappings["Temperature"]
    HUMIDITY_MAP = mappings["Humidity"]
    WIND_MAP = mappings["Wind"]

    scenarios = [
        {"Month": "July", "Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"},
        {"Month": "January", "Outlook": "Overcast", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"},
        {"Month": "April", "Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak"},
        {"Month": "October", "Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong"}
    ]

    for i, sc in enumerate(scenarios, 1):
        sample_df = pd.DataFrame([{
            "Month": MONTH_MAP[sc["Month"]],
            "Outlook": OUTLOOK_MAP[sc["Outlook"]],
            "Temperature": TEMP_MAP[sc["Temperature"]],
            "Humidity": HUMIDITY_MAP[sc["Humidity"]],
            "Wind": WIND_MAP[sc["Wind"]]
        }])

        nb_pred = CLASS_NAMES[nb_model.predict(sample_df)[0]]
        nb_prob = nb_model.predict_proba(sample_df)[0][1] * 100

        dt_pred = CLASS_NAMES[dt_model.predict(sample_df)[0]]
        dt_prob = dt_model.predict_proba(sample_df)[0][1] * 100

        print(f"\n  Scenario {i}: {sc['Month']}, {sc['Outlook']}, {sc['Temperature']}, {sc['Humidity']} Humidity, {sc['Wind']} Wind")
        print(f"    • Naïve Bayes Prediction : {nb_pred:<5}  (Confidence: {nb_prob:.1f}% Yes)")
        print(f"    • Decision Tree Prediction: {dt_pred:<5}  (Confidence: {dt_prob:.1f}% Yes)")

    print("\n" + "=" * 80)
    print("                     LIVE COMPARATIVE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_live_comparative_suite()
