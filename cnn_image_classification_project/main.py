# -*- coding: utf-8 -*-
"""
main.py
-------
End-to-end pipeline runner for Experiment 4: CNN Image Classification.
EEE Domain Application: Solar PV Panel Thermal & Electroluminescence Fault Detection.
Orchestrates: Config loading -> EEE thermography data synthesis & preprocessing ->
data augmentation -> SolarPV_CNN architecture construction -> PyTorch training loop
with early stopping -> test set evaluation -> visualization & report generation.

Usage:
    py main.py
    py main.py --config config/hyperparameters.json
    py main.py --epochs 15
"""

import argparse
import os
import sys
import time

# Reconfigure stdout/stderr for UTF-8 compatibility on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure src/ is importable when running from project root
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from data_loader import load_config, prepare_mnist_dataloaders, explore_dataset_summary
from model_builder import build_cnn_model, summarize_model
from training import train_model
from evaluation import evaluate_model_on_test_set, save_evaluation_csv, generate_classification_report
from visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions,
    plot_misclassified_samples,
    plot_learned_filters,
)


def print_banner() -> None:
    banner = (
        "\n"
        "  +==============================================================+\n"
        "  |   CNN IMAGE CLASSIFICATION - EEE DOMAIN EXPERIMENT           |\n"
        "  |   Application : Solar PV Panel Thermal Fault Detection       |\n"
        "  |   Classes     : Healthy, Micro Crack, Hotspot, Dust/Soiling  |\n"
        "  +==============================================================+\n"
    )
    print(banner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CNN Solar PV Panel Fault Classification Pipeline — EEE Experiment"
    )
    _default_cfg = os.path.join(_PROJECT_ROOT, "config", "hyperparameters.json")
    parser.add_argument(
        "--config", default=_default_cfg,
        help="Path to hyperparameters.json configuration file.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override training batch size.",
    )
    return parser.parse_args()


def main() -> None:
    t_start = time.time()
    args    = parse_args()
    print_banner()

    # -- 1. Load configuration ----------------------------------------------
    print("STEP 1 -> Load configuration")
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    # -- 2. Data loading & preprocessing -----------------------------------
    print("\nSTEP 2 -> Data collection, augmentation & DataLoader preparation")
    train_loader, val_loader, test_loader = prepare_mnist_dataloaders(cfg)
    explore_dataset_summary(train_loader, test_loader)

    # -- 3. Model construction ----------------------------------------------
    print("STEP 3 -> Model selection & architecture design")
    model = build_cnn_model(cfg)
    summarize_model(model)

    # -- 4. Model training --------------------------------------------------
    print("STEP 4 -> Model training loop & validation monitoring")
    history = train_model(model, train_loader, val_loader, cfg)

    # -- 5. Test set evaluation ---------------------------------------------
    print("STEP 5 -> Held-out test set evaluation & metric computation")
    metrics = evaluate_model_on_test_set(model, test_loader, cfg)

    # -- 6. Export evaluation artefacts ------------------------------------
    print("STEP 6 -> Saving evaluation reports & CSV metrics")
    save_evaluation_csv(metrics, cfg["output"]["metrics_csv"])
    generate_classification_report(metrics, cfg, cfg["output"]["report_txt"])

    # -- 7. Generate visualisations -----------------------------------------
    print("\nSTEP 7 -> Generating publication-quality visualisations")
    plot_training_history(history, cfg["output"]["training_history_plot"])
    plot_confusion_matrix(metrics["confusion_matrix"], cfg["output"]["confusion_matrix_plot"])
    plot_sample_predictions(metrics, cfg["output"]["sample_predictions_plot"])
    plot_misclassified_samples(metrics, cfg["output"]["misclassified_samples_plot"])
    plot_learned_filters(model, cfg["output"]["learned_filters_plot"])

    # -- 8. Pipeline Summary ------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print("  CNN EEE DOMAIN IMAGE CLASSIFICATION PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Dataset Name                : {cfg['dataset']['name']}")
    print(f"  Application                 : {cfg['dataset']['application']}")
    print(f"  Test Accuracy               : {metrics['test_accuracy'] * 100:.2f}%")
    print(f"  Test Loss                   : {metrics['test_loss']:.4f}")
    print(f"  Macro F1-Score              : {metrics['f1_macro']:.4f}")
    print(f"  Total Misclassified (Test)  : {len(metrics['misclassified_indices']):,} / {len(metrics['y_true']):,}")
    print(f"  Total Runtime               : {elapsed:.1f}s")
    print(f"\n  Outputs written to          : {cfg['output']['results_dir']}")
    print(f"  Model saved to              : {cfg['output']['model_path']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
