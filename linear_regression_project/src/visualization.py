"""
Visualization Module
====================
Generate publication-quality plots for the Linear Regression project:
feature correlations, scatter / regression lines, residual plots,
actual-vs-predicted comparisons, and cross-validation bar charts.

Functions:
    plot_correlation_heatmap()   – Seaborn heatmap of feature correlations
    plot_feature_distribution()  – Histograms of every feature
    plot_simple_regression()     – Scatter + regression line (single feature)
    plot_actual_vs_predicted()   – 45° reference line for both models
    plot_residuals()             – Residual scatter & histogram
    plot_residual_distribution() – QQ plot + histogram of residuals
    plot_cross_validation()      – Bar chart of CV fold scores
    plot_coefficient_importance()– Horizontal bar of Multiple LR coefficients
    generate_all_plots()         – Create and save every plot
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

def _apply_style(config: dict = None):
    """Apply global matplotlib / seaborn style from config."""
    style = "seaborn-v0_8-whitegrid"
    if config and "visualization" in config:
        style = config["visualization"].get("style", style)
    try:
        plt.style.use(style)
    except Exception:
        plt.style.use("ggplot")
    sns.set_palette("viridis" if config is None else config.get("visualization", {}).get("color_palette", "viridis"))


def _figsize(config: dict = None):
    if config and "visualization" in config:
        return tuple(config["visualization"].get("figure_size", [10, 6]))
    return (10, 6)


def _dpi(config: dict = None):
    if config and "visualization" in config:
        return config["visualization"].get("dpi", 150)
    return 150


def _save(fig, path, config=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=_dpi(config), bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# 1. Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(corr_matrix, save_path, config=None):
    _apply_style(config)
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlBu_r", center=0, linewidths=0.5,
        square=True, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=15)
    _save(fig, save_path, config)


# ---------------------------------------------------------------------------
# 2. Feature Distribution
# ---------------------------------------------------------------------------

def plot_feature_distribution(df, save_path, config=None):
    _apply_style(config)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n = len(numeric_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=40, color=sns.color_palette()[0], edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].axvline(df[col].mean(), color="red", linestyle="--", linewidth=1, label="mean")
        axes[i].axvline(df[col].median(), color="orange", linestyle="-.", linewidth=1, label="median")
        axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_path, config)


# ---------------------------------------------------------------------------
# 3. Simple Regression Line
# ---------------------------------------------------------------------------

def plot_simple_regression(
    X_feature, y_true, y_pred, feature_name, save_path, config=None,
):
    _apply_style(config)
    fig, ax = plt.subplots(figsize=_figsize(config))

    ax.scatter(X_feature, y_true, alpha=0.3, s=10, label="Actual", color="#3498db")
    # Sort for clean line
    order = np.argsort(X_feature.ravel())
    ax.plot(
        X_feature.ravel()[order], y_pred.ravel()[order],
        color="#e74c3c", linewidth=2, label="Regression Line",
    )
    ax.set_xlabel(f"{feature_name} (scaled)", fontsize=13)
    ax.set_ylabel("Median House Value ($100k)", fontsize=13)
    ax.set_title(f"Simple Linear Regression: {feature_name} vs Price", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    _save(fig, save_path, config)


# ---------------------------------------------------------------------------
# 4. Actual vs Predicted
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true, y_pred, model_name, save_path, config=None,
):
    _apply_style(config)
    fig, ax = plt.subplots(figsize=_figsize(config))

    ax.scatter(y_true, y_pred, alpha=0.35, s=12, color="#2ecc71")
    lims = [
        min(min(y_true), min(y_pred)),
        max(max(y_true), max(y_pred)),
    ]
    ax.plot(lims, lims, "--", color="#e74c3c", linewidth=1.5, label="Ideal (y=x)")
    ax.set_xlabel("Actual Values", fontsize=13)
    ax.set_ylabel("Predicted Values", fontsize=13)
    ax.set_title(f"Actual vs. Predicted — {model_name}", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    _save(fig, save_path, config)


# ---------------------------------------------------------------------------
# 5. Residual Plots
# ---------------------------------------------------------------------------

def plot_residuals(y_true, y_pred, model_name, save_path, config=None):
    _apply_style(config)
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter
    axes[0].scatter(y_pred, residuals, alpha=0.35, s=10, color="#9b59b6")
    axes[0].axhline(0, color="#e74c3c", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Predicted Values", fontsize=12)
    axes[0].set_ylabel("Residuals", fontsize=12)
    axes[0].set_title(f"Residuals vs. Predicted — {model_name}", fontsize=14, fontweight="bold")

    # Histogram
    axes[1].hist(residuals, bins=50, color="#1abc9c", edgecolor="white", alpha=0.85, density=True)
    # Overlay normal curve
    mu, sigma = residuals.mean(), residuals.std()
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    axes[1].plot(x_range, stats.norm.pdf(x_range, mu, sigma), color="#e74c3c", linewidth=2, label="Normal fit")
    axes[1].set_xlabel("Residual Value", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_title(f"Residual Distribution — {model_name}", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)

    fig.tight_layout()
    _save(fig, save_path, config)


# ---------------------------------------------------------------------------
# 6. QQ Plot
# ---------------------------------------------------------------------------

def plot_residual_qq(residuals, model_name, save_path, config=None):
    _apply_style(config)
    fig, ax = plt.subplots(figsize=_figsize(config))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f"QQ Plot of Residuals — {model_name}", fontsize=15, fontweight="bold")
    ax.get_lines()[0].set(marker="o", markersize=3, alpha=0.5, color="#3498db")
    ax.get_lines()[1].set(color="#e74c3c", linewidth=2)
    _save(fig, save_path, config)


# ---------------------------------------------------------------------------
# 7. Cross-Validation Scores
# ---------------------------------------------------------------------------

def plot_cross_validation(
    simple_cv_scores, multi_cv_scores, save_path, config=None,
):
    _apply_style(config)
    n_folds = len(simple_cv_scores)
    x = np.arange(1, n_folds + 1)
    width = 0.35

    fig, ax = plt.subplots(figsize=_figsize(config))
    bars1 = ax.bar(x - width / 2, simple_cv_scores, width, label="Simple LR", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width / 2, multi_cv_scores, width, label="Multiple LR", color="#e74c3c", alpha=0.85)

    ax.set_xlabel("Fold", fontsize=13)
    ax.set_ylabel("R² Score", fontsize=13)
    ax.set_title("Cross-Validation R² Scores", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.legend(fontsize=11)

    # Annotate bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save(fig, save_path, config)


# ---------------------------------------------------------------------------
# 8. Coefficient Importance
# ---------------------------------------------------------------------------

def plot_coefficient_importance(coeff_df, save_path, config=None):
    _apply_style(config)
    fig, ax = plt.subplots(figsize=_figsize(config))

    sorted_df = coeff_df.sort_values("Abs_Coefficient")
    colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in sorted_df["Coefficient"]]
    ax.barh(sorted_df["Feature"], sorted_df["Coefficient"], color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Coefficient Value", fontsize=13)
    ax.set_title("Feature Coefficients — Multiple Linear Regression", fontsize=15, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2ecc71", label="Positive"), Patch(facecolor="#e74c3c", label="Negative")]
    ax.legend(handles=legend_elements, fontsize=10)

    fig.tight_layout()
    _save(fig, save_path, config)


# ---------------------------------------------------------------------------
# 9. Generate All Plots
# ---------------------------------------------------------------------------

def generate_all_plots(
    data: dict,
    simple_result: dict,
    multi_result: dict,
    simple_metrics: dict,
    multi_metrics: dict,
    simple_cv: dict,
    multi_cv: dict,
    simple_residuals: dict,
    multi_residuals: dict,
    config: dict,
):
    """
    Orchestrate creation and saving of every visualisation.
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(base, config["output"]["plots_dir"])
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("GENERATING PLOTS")
    print(f"{'=' * 60}")

    # 1. Correlation heatmap
    plot_correlation_heatmap(
        data["corr_matrix"],
        os.path.join(plots_dir, "feature_correlation.png"), config,
    )

    # 2. Feature distributions
    plot_feature_distribution(
        data["df_clean"],
        os.path.join(plots_dir, "feature_distributions.png"), config,
    )

    # 3. Simple regression scatter + line
    fi = simple_result["feature_index"]
    X_feat_test = np.array(data["X_test_scaled"])[:, fi].reshape(-1, 1)
    y_pred_simple_test = simple_result["model"].predict(X_feat_test)
    plot_simple_regression(
        X_feat_test, data["y_test"], y_pred_simple_test,
        simple_result["feature_name"],
        os.path.join(plots_dir, "simple_regression_line.png"), config,
    )

    # 4a. Actual vs predicted – Simple
    plot_actual_vs_predicted(
        data["y_test"], simple_metrics["test"]["predictions"],
        "Simple LR",
        os.path.join(plots_dir, "actual_vs_predicted_simple.png"), config,
    )
    # 4b. Actual vs predicted – Multiple
    plot_actual_vs_predicted(
        data["y_test"], multi_metrics["test"]["predictions"],
        "Multiple LR",
        os.path.join(plots_dir, "actual_vs_predicted_multiple.png"), config,
    )

    # 5a. Residuals – Simple
    plot_residuals(
        data["y_test"], simple_metrics["test"]["predictions"],
        "Simple LR",
        os.path.join(plots_dir, "residual_plot_simple.png"), config,
    )
    # 5b. Residuals – Multiple
    plot_residuals(
        data["y_test"], multi_metrics["test"]["predictions"],
        "Multiple LR",
        os.path.join(plots_dir, "residual_plot_multiple.png"), config,
    )

    # 6. QQ plots
    plot_residual_qq(
        simple_residuals["residuals"], "Simple LR",
        os.path.join(plots_dir, "qq_plot_simple.png"), config,
    )
    plot_residual_qq(
        multi_residuals["residuals"], "Multiple LR",
        os.path.join(plots_dir, "qq_plot_multiple.png"), config,
    )

    # 7. Cross-validation comparison
    plot_cross_validation(
        simple_cv["scores"], multi_cv["scores"],
        os.path.join(plots_dir, "cross_validation_scores.png"), config,
    )

    # 8. Coefficient importance (Multiple LR)
    if "coeff_df" in multi_result:
        plot_coefficient_importance(
            multi_result["coeff_df"],
            os.path.join(plots_dir, "coefficient_importance.png"), config,
        )

    print(f"\n[INFO] All plots saved to {plots_dir}")
