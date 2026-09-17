# -*- coding: utf-8 -*-
"""
visualization.py
----------------
VisualizationModule: Publication-quality time-series diagnostic plots:
1. Training History (Train vs Val Loss curves)
2. Predictions vs. Actual Energy Consumption Trajectory
3. Residual Error Analysis (Histogram, Scatter, Cumulative Error)
4. Autocorrelation Function (ACF) Plot
5. 24-Hour Future Recursive Forecast Trajectory
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def set_plot_style():
    """Apply standard clean plotting styles."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 1.0


def plot_training_history(history: dict, save_path: str = "results/training_history.png") -> None:
    """Plot training and validation MSE loss curves over epochs."""
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, history["train_loss"], label="Training Loss (MSE)", color="#1F77B4", linewidth=2.2, marker="o", markersize=4)
    ax.plot(epochs, history["val_loss"], label="Validation Loss (MSE)", color="#FF7F0E", linewidth=2.2, linestyle="--", marker="s", markersize=4)

    ax.set_title("Vanilla RNN Training & Validation Loss Curves (Energy Forecasting)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved training history plot to '{save_path}'")


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "results/predictions_vs_actual.png") -> None:
    """Plot actual vs. predicted time-series energy consumption trajectories."""
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    timesteps = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(timesteps, y_true, label="Actual Energy Consumption (kWh)", color="#2CA02C", linewidth=2.2, alpha=0.85)
    ax.plot(timesteps, y_pred, label="Vanilla RNN Forecast (kWh)", color="#D62728", linewidth=1.8, linestyle="--")

    ax.set_title("Test Set Hourly Energy Consumption: Actual vs. Vanilla RNN Forecast", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Test Timeline (Hourly Timesteps)", fontsize=11)
    ax.set_ylabel("Energy Consumption (kWh)", fontsize=11)
    ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Annotate summary metrics on plot
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    ax.text(0.98, 0.05, f"RMSE: {rmse:.3f} kWh\nMAE:  {mae:.3f} kWh", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F0F0", edgecolor="#CCCCCC", linewidth=1.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved predictions vs actual plot to '{save_path}'")


def plot_error_analysis(y_true: np.ndarray, y_pred: np.ndarray, residuals: np.ndarray, save_path: str = "results/error_analysis.png") -> None:
    """Generate 3-panel error analysis plot: Residual Histogram, Residual Scatter, Cumulative Error."""
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Residual Histogram & KDE
    ax1.hist(residuals, bins=25, color="#1F77B4", edgecolor="black", alpha=0.7, density=True)
    ax1.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax1.set_title("Residual Error Distribution (kWh)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Residual Error (Actual - Predicted)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Actual vs Residual Scatter
    ax2.scatter(y_true, residuals, color="#9467BD", alpha=0.7, edgecolors="none", s=30)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.set_title("Residuals vs. Actual Energy (kWh)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Actual Energy Consumption (kWh)", fontsize=10)
    ax2.set_ylabel("Residual Error (kWh)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Cumulative Absolute Error
    cum_error = np.cumsum(np.abs(residuals))
    ax3.plot(cum_error, color="#E377C2", linewidth=2.0)
    ax3.set_title("Cumulative Absolute Error", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Test Timesteps (Hours)", fontsize=10)
    ax3.set_ylabel("Cumulative Error (kWh)", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved error analysis plot to '{save_path}'")


def plot_autocorrelation(acf_residuals: np.ndarray, save_path: str = "results/autocorrelation_plot.png") -> None:
    """Plot Autocorrelation Function (ACF) of prediction residuals."""
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lags = np.arange(len(acf_residuals))
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot stem lines
    markerline, stemlines, baseline = ax.stem(lags, acf_residuals)
    plt.setp(stemlines, "color", "#1F77B4", "linewidth", 1.8)
    plt.setp(markerline, "color", "#1F77B4", "markersize", 5)
    plt.setp(baseline, "color", "black", "linewidth", 1.0)

    # 95% Confidence interval bands for white noise (1.96 / sqrt(N))
    conf_bound = 1.96 / np.sqrt(len(acf_residuals) * 3)
    ax.axhline(conf_bound, color="red", linestyle="--", alpha=0.7, label="95% Confidence Interval")
    ax.axhline(-conf_bound, color="red", linestyle="--", alpha=0.7)

    ax.set_title("Autocorrelation Function (ACF) of Residual Errors", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Lag Step (Hours)", fontsize=11)
    ax.set_ylabel("Autocorrelation Coefficient", fontsize=11)
    ax.set_ylim(-0.4, 1.1)
    ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved autocorrelation plot to '{save_path}'")


def plot_future_forecast(
    historical_energy: np.ndarray,
    future_preds: np.ndarray,
    save_path: str = "results/forecast_future.png"
) -> None:
    """Plot 24-hour future recursive multi-step forecasting trajectory."""
    set_plot_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    history_len = len(historical_energy)
    hist_timesteps = np.arange(history_len)
    future_timesteps = np.arange(history_len, history_len + len(future_preds))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Show last 72 hours of historical context
    display_history = min(72, history_len)
    ax.plot(hist_timesteps[-display_history:], historical_energy[-display_history:], label=f"Historical Observed Load (Last {display_history} Hours)", color="#1F77B4", linewidth=2.0)
    ax.plot(future_timesteps, future_preds, label="Vanilla RNN 24-Hour Future Forecast", color="#D62728", linewidth=2.2, linestyle="--", marker="o", markersize=4)

    ax.axvline(history_len - 1, color="gray", linestyle=":", linewidth=1.5, label="Forecast Horizon Boundary")

    ax.set_title("24-Hour Multi-Step Future Energy Consumption Recursive Forecast", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Timeline (Hours)", fontsize=11)
    ax.set_ylabel("Energy Consumption (kWh)", fontsize=11)
    ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved future forecast plot to '{save_path}'")


if __name__ == "__main__":
    dummy_t = np.sin(np.linspace(0, 10, 100)) * 2.0 + 3.0
    dummy_p = dummy_t + np.random.normal(0, 0.2, 100)
    plot_predictions_vs_actual(dummy_t, dummy_p, "results/test_pred.png")

