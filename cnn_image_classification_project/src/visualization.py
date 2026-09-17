"""
visualization.py
----------------
VisualizationModule: Generates publication-quality plots for EEE Solar PV Fault classification:
  1. Training & Validation History (Loss & Accuracy curves)
  2. Confusion Matrix Heatmap
  3. Sample Predictions Grid with Confidence Scores
  4. Misclassified Failure Cases Analysis
  5. Learned First-Layer Convolutional Filters
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = ["Healthy_Panel", "Micro_Crack", "Hotspot_Fault", "Dust_Soiling"]

# Global Clean White Theme Styling
BG_WHITE = "#FFFFFF"
FG_DARK  = "#222222"
GRID_CLR = "#E0E0E0"

plt.rcParams.update({
    "figure.facecolor":  BG_WHITE,
    "axes.facecolor":    BG_WHITE,
    "axes.edgecolor":    GRID_CLR,
    "axes.labelcolor":   FG_DARK,
    "axes.titlecolor":   FG_DARK,
    "text.color":        FG_DARK,
    "xtick.color":       FG_DARK,
    "ytick.color":       FG_DARK,
    "grid.color":        GRID_CLR,
    "grid.linewidth":    0.5,
    "legend.facecolor":  BG_WHITE,
    "legend.edgecolor":  GRID_CLR,
    "font.family":       "DejaVu Sans",
    "figure.dpi":        150,
})


def plot_training_history(history: dict, save_path: str = "results/training_history.png") -> None:
    """Plot dual-panel training and validation loss and accuracy history."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss panel
    ax1 = axes[0]
    ax1.plot(epochs, history["train_loss"], "o-", color="#6C63FF", linewidth=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "s--", color="#FF6584", linewidth=2, label="Val Loss")
    ax1.set_title("Solar PV Fault CNN Loss Dynamics", fontsize=13, pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_xticks(list(epochs)[::max(1, len(epochs)//10)])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.5)

    # Accuracy panel
    ax2 = axes[1]
    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct   = [a * 100 for a in history["val_acc"]]
    ax2.plot(epochs, train_acc_pct, "o-", color="#43CBFF", linewidth=2, label="Train Acc")
    ax2.plot(epochs, val_acc_pct,   "s--", color="#F7971E", linewidth=2, label="Val Acc")
    ax2.set_title("Solar PV Fault CNN Accuracy Dynamics", fontsize=13, pad=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_xticks(list(epochs)[::max(1, len(epochs)//10)])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.5)

    fig.suptitle("Solar PV Panel Fault Detection (EEE Domain) — Training History", fontsize=15, y=1.02, color=FG_DARK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Training history plot saved → {save_path}")


def plot_confusion_matrix(cm: np.ndarray, save_path: str = "results/confusion_matrix.png") -> None:
    """Plot confusion matrix heatmap with EEE domain class names."""
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        linewidths=0.5, linecolor="#E0E0E0"
    )

    ax.set_title("Confusion Matrix — Solar PV Panel Fault Classification", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Fault Class", fontsize=11)
    ax.set_ylabel("True Fault Class", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Confusion matrix heatmap saved → {save_path}")


def plot_sample_predictions(metrics: dict, save_path: str = "results/sample_predictions.png", n_samples: int = 16) -> None:
    """Plot a 4x4 grid of sample test thermography predictions with class probabilities."""
    x_images = metrics["x_images"]
    y_true   = metrics["y_true"]
    y_pred   = metrics["y_pred"]
    y_probs  = metrics["y_probs"]

    fig, axes = plt.subplots(4, 4, figsize=(11, 11))

    for i, ax in enumerate(axes.flat):
        if i >= len(x_images) or i >= n_samples:
            ax.axis("off")
            continue

        img = x_images[i].squeeze()
        true_lbl = y_true[i]
        pred_lbl = y_pred[i]
        conf     = y_probs[i][pred_lbl] * 100

        ax.imshow(img, cmap="inferno")
        color = "#2E7D32" if true_lbl == pred_lbl else "#C62828"
        ax.set_title(f"Pred: {CLASS_NAMES[pred_lbl]} ({conf:.1f}%)\nTrue: {CLASS_NAMES[true_lbl]}", fontsize=8, color=color)
        ax.axis("off")

    fig.suptitle("Solar PV Test Predictions & Thermal Thermography Features", fontsize=14, y=1.02, color=FG_DARK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Sample predictions grid saved → {save_path}")


def plot_misclassified_samples(metrics: dict, save_path: str = "results/misclassified_samples.png", n_samples: int = 16) -> None:
    """Plot grid of misclassified failure cases with true vs predicted labels."""
    mis_idx  = metrics["misclassified_indices"]
    x_images = metrics["x_images"]
    y_true   = metrics["y_true"]
    y_pred   = metrics["y_pred"]
    y_probs  = metrics["y_probs"]

    if len(mis_idx) == 0:
        print("[Visualization] No misclassifications found — skipping failure cases plot.")
        return

    n_display = min(len(mis_idx), n_samples)
    rows = int(np.ceil(n_display / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(11, 2.8 * rows))

    axes_flat = axes.flat if isinstance(axes, np.ndarray) else [axes]

    for i, ax in enumerate(axes_flat):
        if i >= n_display:
            ax.axis("off")
            continue

        idx      = mis_idx[i]
        img      = x_images[idx].squeeze()
        true_lbl = y_true[idx]
        pred_lbl = y_pred[idx]
        conf     = y_probs[idx][pred_lbl] * 100

        ax.imshow(img, cmap="inferno")
        ax.set_title(f"Pred: {CLASS_NAMES[pred_lbl]} ({conf:.1f}%)\nTrue: {CLASS_NAMES[true_lbl]}", fontsize=8, color="#C62828")
        ax.axis("off")

    fig.suptitle("Analysis of Misclassified Solar PV Fault Samples", fontsize=14, y=1.02, color=FG_DARK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Misclassified samples plot saved → {save_path}")


def plot_learned_filters(model, save_path: str = "results/learned_filters.png") -> None:
    """Visualize 32 learned 3x3 filters from Conv1 layer."""
    try:
        conv1_weights = model.conv1.weight.detach().cpu().numpy() # (32, 1, 3, 3)
    except Exception as e:
        print(f"[Visualization] Could not extract Conv1 weights: {e}")
        return

    n_filters = conv1_weights.shape[0]
    fig, axes = plt.subplots(4, 8, figsize=(10, 5))

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            filt = conv1_weights[i, 0]
            ax.imshow(filt, cmap="viridis", interpolation="nearest")
            ax.set_title(f"F{i+1}", fontsize=8)
        ax.axis("off")

    fig.suptitle("Learned Conv1 Filters (Solar Thermography Spatial Edge & Hotspot Feature Extractor)", fontsize=11, y=1.02, color=FG_DARK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Learned filters plot saved → {save_path}")


if __name__ == "__main__":
    print("[Visualization] Module compiled successfully.")
