# -*- coding: utf-8 -*-
"""
main.py
-------
End-to-end pipeline runner for Experiment 5: Sequential Data Modeling using Vanilla RNN.
Orchestrates: Config loading → Dataset loading/generation → MinMaxScaler normalization →
sliding lookback window creation → PyTorch Stacked Vanilla RNN construction → training loop
with early stopping → test set evaluation → recursive 24-hour future forecasting →
visualization & report generation.

Usage:
    py main.py
    py main.py --config config/hyperparameters.json
    py main.py --epochs 30
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

# Ensure src/ is importable when running from project root
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from data_loader import load_config, get_or_create_energy_data, explore_time_series_summary
from time_series_preprocessing import TimeSeriesScaler, clean_and_interpolate_series, verify_preprocessing_quality
from sequence_generator import create_sequences, split_time_series_sequences, prepare_pytorch_dataloaders
from model_builder import build_rnn_model, summarize_model
from training import train_model
from evaluation import (
    evaluate_model_on_test_set,
    forecast_future_timesteps,
    save_predictions_csv,
    generate_metrics_report,
)
from visualization import (
    plot_training_history,
    plot_predictions_vs_actual,
    plot_error_analysis,
    plot_autocorrelation,
    plot_future_forecast,
)


def print_banner() -> None:
    banner = (
        "\n"
        "  +==============================================================+\n"
        "  |        VANILLA RNN SEQUENTIAL TIME-SERIES MODELING           |\n"
        "  |   Experiment 5: Household Energy Consumption Forecasting     |\n"
        "  |      Dataset: 2,160 Hourly Observations (Window=24 hrs)      |\n"
        "  +==============================================================+\n"
    )
    print(banner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vanilla RNN Energy Consumption Forecasting Pipeline — Experiment 5"
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
    args = parse_args()
    print_banner()

    # ── 1. Load configuration ──────────────────────────────────────────────
    print("STEP 1 [>] Load configuration & set hyperparameters")
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size

    seed = cfg["training"].get("random_seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    target_col = cfg["dataset"].get("target_column", "Energy_Consumption_kWh")
    lookback   = cfg["sequence"].get("lookback_window", 24)

    # ── 2. Data Acquisition & EDA ─────────────────────────────────────────
    print("\nSTEP 2 [>] Load hourly household energy consumption time-series dataset")
    df = get_or_create_energy_data(cfg)
    explore_time_series_summary(df, target_col=target_col)

    # Clean & interpolate missing steps if any
    raw_energy_series = clean_and_interpolate_series(df[target_col])
    raw_energy_np = raw_energy_series.values

    # ── 3. Data Preprocessing & MinMaxScaler Normalization ────────────────
    print("STEP 3 [>] Time-series preprocessing & MinMaxScaler normalization [0, 1]")
    scaler = TimeSeriesScaler(feature_range=cfg["preprocessing"].get("feature_range", [0, 1]))
    scaled_energy_np = scaler.fit_transform(raw_energy_np)
    verify_preprocessing_quality(raw_energy_np, scaled_energy_np)

    # ── 4. Sliding Window Sequence Generation & Dataset Split ──────────────
    print("STEP 4 [>] Generate sliding lookback window sequences & temporal dataset split")
    X_seq, y_seq = create_sequences(scaled_energy_np, lookback=lookback, lookahead=1)
    
    train_ratio = cfg["dataset"].get("train_split", 0.8)
    val_ratio   = cfg["dataset"].get("val_split", 0.1)

    X_tr, y_tr, X_val, y_val, X_te, y_te = split_time_series_sequences(
        X_seq, y_seq, train_ratio=train_ratio, val_ratio=val_ratio
    )

    batch_size = cfg["training"].get("batch_size", 32)
    train_loader, val_loader, test_loader = prepare_pytorch_dataloaders(
        X_tr, y_tr, X_val, y_val, X_te, y_te, batch_size=batch_size
    )

    # ── 5. Stacked Vanilla RNN Model Construction ─────────────────────────
    print("\nSTEP 5 [>] Stacked Vanilla RNN network architecture construction")
    model = build_rnn_model(cfg)
    summarize_model(model)

    # ── 6. Model Training & Validation ────────────────────────────────────
    print("STEP 6 [>] Model training loop (BPTT) with Adam optimizer & early stopping")
    history = train_model(model, train_loader, val_loader, cfg)

    # ── 7. Test Set Evaluation & Inverse Scaling ──────────────────────────
    print("\nSTEP 7 [>] Held-out test set evaluation & inverse scaling to original kWh units")
    metrics = evaluate_model_on_test_set(model, test_loader, scaler, cfg)

    # ── 8. 24-Hour Recursive Future Forecasting ───────────────────────────
    print("STEP 8 [>] Multi-step 24-hour recursive future energy consumption forecasting")
    last_window_scaled = scaled_energy_np[-lookback:]  # Last 24 hours
    future_forecast = forecast_future_timesteps(model, last_window_scaled, scaler, future_steps=24)
    print(f"  - Forecasted next 24 hours mean energy : {np.mean(future_forecast):.3f} kWh")
    print(f"  - Forecasted energy range [min, max]   : [{np.min(future_forecast):.3f}, {np.max(future_forecast):.3f}] kWh")

    # ── 9. Export Evaluation Artefacts ────────────────────────────────────
    print("\nSTEP 9 [>] Saving evaluation reports & CSV predictions")
    save_predictions_csv(metrics["y_true"], metrics["y_pred"], cfg["output"]["predictions_csv"])
    generate_metrics_report(metrics, cfg, cfg["output"]["metrics_report_txt"])

    # ── 10. Generate Visualisations ────────────────────────────────────────
    print("\nSTEP 10 [>] Generating 5 publication-quality time-series diagnostic plots")
    plot_training_history(history, cfg["output"]["training_history_plot"])
    plot_predictions_vs_actual(metrics["y_true"], metrics["y_pred"], cfg["output"]["predictions_plot"])
    plot_error_analysis(metrics["y_true"], metrics["y_pred"], metrics["residuals"], cfg["output"]["error_analysis_plot"])
    plot_autocorrelation(metrics["acf_residuals"], cfg["output"]["autocorrelation_plot"])
    plot_future_forecast(raw_energy_np, future_forecast, cfg["output"]["forecast_future_plot"])

    # ── 11. Pipeline Summary ───────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print("  VANILLA RNN ENERGY CONSUMPTION FORECASTING PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Root Mean Squared Error (RMSE) : {metrics['rmse']:.4f} kWh")
    print(f"  Mean Absolute Error (MAE)      : {metrics['mae']:.4f} kWh")
    print(f"  Mean Absolute Percentage (MAPE): {metrics['mape']:.2f}%")
    print(f"  R² Forecast Determination Score: {metrics['r2']:.4f}")
    print(f"  Total Runtime                  : {elapsed:.1f}s")
    print(f"\n  Outputs written to             : {cfg['output']['results_dir']}")
    print(f"  Model saved to                 : {cfg['output']['model_path']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

