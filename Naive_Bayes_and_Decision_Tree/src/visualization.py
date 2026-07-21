"""
visualization.py
================
VisualizationModule: Plot confusion matrices, ROC curves, decision tree, and
feature importance.

Functions
---------
plot_confusion_matrices(results, class_names, save_path)
    Side-by-side heatmaps for all models.
plot_roc_curves(roc_data_list, save_path)
    Overlaid ROC curves for all models.
plot_decision_tree(model, feature_names, class_names, save_path)
    Tree diagram using sklearn.tree.plot_tree.
plot_feature_importance(model, feature_names, save_path)
    Horizontal bar chart of DT feature importances.
plot_depth_sweep(records, save_path)
    Train vs test accuracy across tree depths.
plot_cv_comparison(cv_results, save_path)
    Box-plot of cross-validation scores per model.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in notebooks too
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.tree import plot_tree


# Consistent colour palette
_PALETTE = {
    "NB":  "#4C72B0",   # steel blue
    "DT":  "#DD8452",   # warm orange
    "pos": "#2ecc71",   # green  (correct)
    "neg": "#e74c3c",   # red    (error)
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
})


# ──────────────────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Confusion matrices
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    results: list,
    class_names: list = None,
    save_path: str = "results/confusion_matrices.png",
) -> None:
    """Side-by-side confusion matrix heatmaps.

    Parameters
    ----------
    results : list of dicts from evaluate_classification() — must have
              'model_name' and 'confusion_matrix' keys.
    class_names : list of str, e.g. ['No', 'Yes']
    save_path : str
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = res["confusion_matrix"]
        labels = class_names if class_names else [str(i) for i in range(cm.shape[0])]

        # Normalised values for annotation
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        annot = np.array(
            [[f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)"
              for j in range(cm.shape[1])]
             for i in range(cm.shape[0])]
        )

        sns.heatmap(
            cm_norm, annot=annot, fmt="", ax=ax,
            cmap="Blues", linewidths=0.5, linecolor="#cccccc",
            xticklabels=labels, yticklabels=labels,
            vmin=0, vmax=1, cbar_kws={"label": "Normalised proportion"},
        )
        ax.set_title(f"{res['model_name']}\nAccuracy: {res['accuracy']:.4f}",
                     fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrices saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# ROC Curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    roc_data_list: list,
    save_path: str = "results/roc_curves.png",
) -> None:
    """Overlaid ROC curves for multiple models.

    Parameters
    ----------
    roc_data_list : list of dicts  {model_name, fpr, tpr, auc}
    save_path : str
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    colours = list(_PALETTE.values())
    for i, rdata in enumerate(roc_data_list):
        if rdata is None:
            continue
        colour = colours[i % len(colours)]
        ax.plot(
            rdata["fpr"], rdata["tpr"],
            color=colour, lw=2.5,
            label=f"{rdata['model_name']}  (AUC = {rdata['auc']:.3f})",
        )
        # Shade under curve
        ax.fill_between(rdata["fpr"], rdata["tpr"], alpha=0.08, color=colour)

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("ROC Curves — Model Comparison", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.85)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Optimal threshold marker (Youden's J)
    for rdata in roc_data_list:
        if rdata is None:
            continue
        j = rdata["tpr"] - rdata["fpr"]
        opt_idx = np.argmax(j)
        ax.scatter(
            rdata["fpr"][opt_idx], rdata["tpr"][opt_idx],
            s=80, zorder=5, marker="*",
            label=f"Optimal ({rdata['model_name']})",
        )

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Decision Tree visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_decision_tree(
    model,
    feature_names: list,
    class_names: list = None,
    save_path: str = "results/decision_tree_visualization.png",
) -> None:
    """Render the fitted Decision Tree using sklearn's plot_tree.

    Parameters
    ----------
    model : fitted DecisionTreeClassifier
    feature_names : list of str
    class_names : list of str, optional
    save_path : str
    """
    depth = model.get_depth()
    # Scale figure height by depth
    fig_w = max(14, depth * 4)
    fig_h = max(8, depth * 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True,
        precision=3,
        ax=ax,
        fontsize=9,
    )
    ax.set_title(
        f"Decision Tree Visualization\n"
        f"Depth={depth}  |  Leaves={model.get_n_leaves()}",
        fontsize=14, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Decision tree visualization saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Feature importance
# ──────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list,
    save_path: str = "results/feature_importance.png",
) -> None:
    """Horizontal bar chart of Decision Tree feature importances (Gini)."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    colours = [
        _PALETTE["DT"] if imp > 0 else "#cccccc"
        for imp in importances[sorted_idx]
    ]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.55)))
    bars = ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color=colours, edgecolor="white", height=0.6,
    )
    # Annotate values
    for bar, val in zip(bars, importances[sorted_idx]):
        if val > 0.01:
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Feature Importance (Gini Impurity Reduction)")
    ax.set_title("Decision Tree — Feature Importance", fontweight="bold")
    ax.set_xlim(0, max(importances) * 1.2)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance chart saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Depth sweep plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_depth_sweep(
    records: list,
    save_path: str = "results/depth_sweep.png",
) -> None:
    """Train vs test accuracy across tree depths.

    Parameters
    ----------
    records : list of dict  [{depth, train_acc, test_acc}, ...]
    save_path : str
    """
    depths      = [r["depth"]     for r in records]
    train_accs  = [r["train_acc"] for r in records]
    test_accs   = [r["test_acc"]  for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, train_accs, "o-", color=_PALETTE["NB"], lw=2, label="Train Accuracy")
    ax.plot(depths, test_accs,  "s-", color=_PALETTE["DT"], lw=2, label="Test Accuracy")

    best_depth = depths[np.argmax(test_accs)]
    ax.axvline(best_depth, color="gray", linestyle="--", lw=1,
               label=f"Best test depth = {best_depth}")

    ax.set_xlabel("Max Tree Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Decision Tree Depth Sweep\n(Overfitting Analysis)", fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xticks(depths)
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Depth sweep plot saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_cv_comparison(
    cv_results: list,
    save_path: str = "results/cv_comparison.png",
) -> None:
    """Box-plot comparing cross-validation score distributions.

    Parameters
    ----------
    cv_results : list of dicts from cross_validate_model()
                 Must have 'model_name' and 'cv_scores' keys.
    save_path : str
    """
    names  = [r["model_name"] for r in cv_results]
    scores = [r["cv_scores"]  for r in cv_results]
    colours = [_PALETTE["NB"], _PALETTE["DT"]][:len(names)]

    fig, ax = plt.subplots(figsize=(7, 5))
    import matplotlib
    _mpl_ver = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
    _bp_labels_key = "tick_labels" if _mpl_ver >= (3, 9) else "labels"
    bp = ax.boxplot(
        scores, patch_artist=True,
        widths=0.4, medianprops={"color": "black", "linewidth": 2},
        **{_bp_labels_key: names},
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    # Overlay individual fold points
    for i, (score_list, colour) in enumerate(zip(scores, colours), start=1):
        jitter = np.random.default_rng(0).uniform(-0.07, 0.07, len(score_list))
        ax.scatter(
            np.full(len(score_list), i) + jitter, score_list,
            color=colour, s=40, zorder=3, edgecolors="white", linewidths=0.5,
        )

    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-Validation Score Comparison\n(Stratified k-Fold)", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  CV comparison plot saved → {save_path}")
