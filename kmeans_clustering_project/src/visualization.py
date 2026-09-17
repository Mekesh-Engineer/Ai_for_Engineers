"""
visualization.py
-----------------
VisualizationModule: Produces publication-quality plots for the K-Means
clustering experiment:
  1. Elbow Curve       – Inertia vs. k
  2. Silhouette Plot   – Average score per k + per-sample bar chart
  3. Gap Statistic     – Gap values with error bars
  4. 2-D Cluster Plot  – PCA projection with cluster boundaries
  5. 3-D Cluster Plot  – PCA 3-component scatter
  6. Pair Plot         – Feature-pair scatter coloured by cluster
  7. Radar / Spider Chart – Cluster centers profile
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
# Only force the non-interactive Agg backend when running as a plain script.
# In Jupyter (ipykernel present) we leave the backend alone so that
# %matplotlib inline / widget backends work correctly.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3-D projection)
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score


# ── Global style ──────────────────────────────────────────────────────────────
PALETTE  = ["#6C63FF", "#FF6584", "#43CBFF", "#F7971E", "#a8ff78",
            "#f953c6", "#4facfe", "#00f2fe", "#43e97b", "#fa709a"]
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

# NOTE: results/ directory is created on-demand by each save function;
# no module-level makedirs here (it would use the wrong CWD in Jupyter).


# ─────────────────────────────────────────────────────────────────────────────
#  1. Elbow Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_elbow_curve(
    k_values: list,
    inertias: list,
    optimal_k: int,
    save_path: str = "results/elbow_plot.png",
) -> None:
    """Plot inertia vs. k with the optimal k highlighted."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(k_values, inertias, "o-", color=PALETTE[0], linewidth=2.5,
            markersize=8, markerfacecolor="#FF6584", markeredgewidth=1.5,
            markeredgecolor="white", label="Inertia")

    # Annotate optimal k
    idx = k_values.index(optimal_k)
    ax.axvline(x=optimal_k, color=PALETTE[1], linestyle="--",
               linewidth=1.8, alpha=0.8, label=f"Optimal k = {optimal_k}")
    ax.scatter([optimal_k], [inertias[idx]], s=200, zorder=5,
               color="#FF6584", edgecolors="white", linewidth=2)
    ax.annotate(f"  k = {optimal_k}\n  Inertia = {inertias[idx]:.2f}",
                xy=(optimal_k, inertias[idx]),
                xytext=(optimal_k + 0.4, inertias[idx] * 1.05),
                color=FG_DARK, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=PALETTE[1]))

    # Fill under curve
    ax.fill_between(k_values, inertias, alpha=0.15, color=PALETTE[0])

    ax.set_title("Elbow Method — Optimal Number of Clusters", fontsize=14, pad=14)
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia  (Within-Cluster SS)", fontsize=12)
    ax.set_xticks(k_values)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Elbow plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  2. Silhouette Plot (average score + per-sample bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_silhouette_scores(
    k_values: list,
    silhouette_scores: list,
    X: np.ndarray,
    labels: np.ndarray,
    optimal_k: int,
    save_path: str = "results/silhouette_plot.png",
) -> None:
    """Dual-panel silhouette visualisation: average bar + per-sample subplot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: average scores per k ──
    ax = axes[0]
    colors = [PALETTE[1] if k == optimal_k else PALETTE[0] for k in k_values]
    bars = ax.bar(k_values, silhouette_scores, color=colors, edgecolor="white",
                  linewidth=0.6, width=0.6, alpha=0.9)
    ax.set_title("Average Silhouette Score per k", fontsize=13, pad=12)
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_xticks(k_values)
    ax.axhline(y=silhouette_scores[k_values.index(optimal_k)],
               color=PALETTE[1], linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Best k={optimal_k}: {silhouette_scores[k_values.index(optimal_k)]:.3f}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.4)
    for bar, score in zip(bars, silhouette_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{score:.3f}", ha="center", va="bottom", fontsize=8.5, color=FG_DARK)

    # ── Right: per-sample silhouette for optimal k ──
    ax2 = axes[1]
    sample_sil = silhouette_samples(X, labels)
    k          = optimal_k
    y_lower    = 10

    for i in range(k):
        cluster_sil = np.sort(sample_sil[labels == i])
        size_i      = len(cluster_sil)
        y_upper     = y_lower + size_i
        color       = PALETTE[i % len(PALETTE)]
        ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                          facecolor=color, edgecolor=color, alpha=0.85)
        ax2.text(-0.05, y_lower + size_i / 2, f"C{i}", fontsize=9,
                 color=FG_DARK, va="center")
        y_lower = y_upper + 10

    ax2.axvline(x=silhouette_score(X, labels), color=FG_DARK,
                linestyle="--", linewidth=1.5, label="Average")
    ax2.set_title(f"Per-Sample Silhouette (k={k})", fontsize=13, pad=12)
    ax2.set_xlabel("Silhouette Coefficient", fontsize=11)
    ax2.set_ylabel("Samples (grouped by cluster)", fontsize=11)
    ax2.set_yticks([])
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="x", alpha=0.4)

    fig.suptitle("Silhouette Analysis", fontsize=15, y=1.02, color=FG_DARK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Silhouette plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  3. Gap Statistic Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_gap_statistic(
    k_values: list,
    gaps: list,
    sdk: list,
    optimal_k: int,
    save_path: str = "results/gap_statistic_plot.png",
) -> None:
    """Plot gap statistic with error bars."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(k_values, gaps, yerr=sdk, fmt="o-",
                color=PALETTE[2], ecolor=PALETTE[1], elinewidth=1.5,
                capsize=5, linewidth=2.5, markersize=8,
                markerfacecolor=PALETTE[3], markeredgecolor="white")

    ax.axvline(x=optimal_k, color=PALETTE[1], linestyle="--",
               linewidth=1.8, alpha=0.8, label=f"Optimal k = {optimal_k}")
    ax.fill_between(k_values, gaps, alpha=0.12, color=PALETTE[2])

    ax.set_title("Gap Statistic — Optimal Number of Clusters", fontsize=14, pad=14)
    ax.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax.set_ylabel("Gap Statistic", fontsize=12)
    ax.set_xticks(k_values)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Gap statistic plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  4. 2-D Cluster Visualisation (PCA)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    centers_scaled: np.ndarray,
    feature_cols: list,
    optimal_k: int,
    save_path: str = "results/cluster_visualization_2d.png",
) -> None:
    """PCA 2-D scatter plot with cluster centroids and decision boundaries."""
    pca   = PCA(n_components=2, random_state=42)
    X_2d  = pca.fit_transform(X)
    ev    = pca.explained_variance_ratio_
    c_2d  = pca.transform(centers_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Decision boundary mesh
    h  = 0.04
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid_pts   = np.c_[xx.ravel(), yy.ravel()]
    # Nearest centroid assignment in PCA space
    dists      = np.linalg.norm(grid_pts[:, None, :] - c_2d[None, :, :], axis=2)
    Z          = dists.argmin(axis=1).reshape(xx.shape)

    cmap_bg = matplotlib.colors.ListedColormap(
        [matplotlib.colors.to_rgba(PALETTE[i], alpha=0.18) for i in range(optimal_k)]
    )
    ax.contourf(xx, yy, Z, cmap=cmap_bg, levels=range(optimal_k + 1))
    ax.contour(xx, yy, Z, colors="white", linewidths=0.4, alpha=0.35)

    # Scatter – data points
    for i in range(optimal_k):
        mask = labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=60, alpha=0.88,
                   color=PALETTE[i], edgecolors="white", linewidth=0.4,
                   label=f"Cluster {i}  (n={mask.sum()})", zorder=3)

    # Centroids
    ax.scatter(c_2d[:, 0], c_2d[:, 1], s=280, marker="*",
               color="white", edgecolors="#222", linewidth=0.8,
               zorder=5, label="Centroids")

    ax.set_title(
        f"K-Means Clusters — PCA 2D Projection  (k={optimal_k})\n"
        f"Explained variance: PC1={ev[0]:.1%}  PC2={ev[1]:.1%}",
        fontsize=13, pad=12,
    )
    ax.set_xlabel(f"Principal Component 1  ({ev[0]:.1%})", fontsize=11)
    ax.set_ylabel(f"Principal Component 2  ({ev[1]:.1%})", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] 2D cluster plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  5. 3-D Cluster Visualisation (PCA)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_clusters_3d(
    X: np.ndarray,
    labels: np.ndarray,
    centers_scaled: np.ndarray,
    optimal_k: int,
    save_path: str = "results/cluster_visualization_3d.png",
) -> None:
    """PCA 3-component 3-D scatter plot."""
    pca   = PCA(n_components=3, random_state=42)
    X_3d  = pca.fit_transform(X)
    c_3d  = pca.transform(centers_scaled)
    ev    = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG_WHITE)
    fig.patch.set_facecolor(BG_WHITE)

    for i in range(optimal_k):
        mask = labels == i
        ax.scatter(
            X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
            s=55, alpha=0.80, color=PALETTE[i],
            edgecolors="white", linewidth=0.25,
            label=f"Cluster {i}  (n={mask.sum()})",
        )

    ax.scatter(c_3d[:, 0], c_3d[:, 1], c_3d[:, 2],
               s=350, marker="*", color="white",
               edgecolors="#111", linewidth=0.8, zorder=5, label="Centroids")

    ax.set_title(
        f"K-Means Clusters — PCA 3D Projection  (k={optimal_k})\n"
        f"Total explained variance: {sum(ev[:3]):.1%}",
        fontsize=12, pad=14, color=FG_DARK,
    )
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=9, labelpad=6)
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=9, labelpad=6)
    ax.set_zlabel(f"PC3 ({ev[2]:.1%})", fontsize=9, labelpad=6)
    ax.tick_params(colors=FG_DARK, labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_CLR)
    ax.yaxis.pane.set_edgecolor(GRID_CLR)
    ax.zaxis.pane.set_edgecolor(GRID_CLR)
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] 3D cluster plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  6. Feature Pair Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_pairs(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list,
    optimal_k: int,
    save_path: str = "results/feature_pair_plot.png",
) -> None:
    """Seaborn pair plot coloured by cluster assignment."""
    df_plot             = df_clean[feature_cols].copy()
    df_plot["Cluster"]  = [f"Cluster {l}" for l in labels]

    cluster_palette = {f"Cluster {i}": PALETTE[i] for i in range(optimal_k)}

    g = sns.pairplot(
        df_plot,
        hue="Cluster",
        palette=cluster_palette,
        plot_kws={"alpha": 0.72, "s": 35, "edgecolor": "white", "linewidth": 0.2},
        diag_kws={"alpha": 0.60},
        corner=False,
    )
    g.figure.suptitle(
        f"Feature Pair Plot — Coloured by K-Means Cluster (k={optimal_k})",
        y=1.02, fontsize=13, color=FG_DARK,
    )
    g.figure.set_facecolor(BG_WHITE)
    for ax in g.axes.flatten():
        if ax:
            ax.set_facecolor(BG_WHITE)
            ax.tick_params(colors=FG_DARK, labelsize=7)
            ax.xaxis.label.set_color(FG_DARK)
            ax.yaxis.label.set_color(FG_DARK)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    g.savefig(save_path, bbox_inches="tight", facecolor=BG_WHITE)
    plt.close("all")
    print(f"[Visualization] Feature pair plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  7. Radar Chart (Cluster Profiles)
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar_chart(
    centers_original: pd.DataFrame,
    feature_cols: list,
    optimal_k: int,
    save_path: str = "results/cluster_radar_chart.png",
) -> None:
    """Spider / radar chart showing normalised cluster center profiles."""
    # FancyArrowPatch import removed — was unused (Bug #11)

    # Normalise centers to [0, 1] per feature for a fair comparison
    mins  = centers_original.min()
    maxs  = centers_original.max()
    denom = (maxs - mins).replace(0, 1)
    normed = (centers_original - mins) / denom

    angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_facecolor(BG_WHITE)
    fig.patch.set_facecolor(BG_WHITE)

    for i in range(optimal_k):
        values = normed.iloc[i].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", color=PALETTE[i],
                linewidth=2.2, markersize=6, label=f"Cluster {i}")
        ax.fill(angles, values, color=PALETTE[i], alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [f.replace("_", "\n") for f in feature_cols],
        size=10, color=FG_DARK,
    )
    ax.set_yticklabels([])
    ax.set_title(
        f"Cluster Center Profiles  (k={optimal_k})\n"
        "(normalised to [0, 1] per feature)",
        size=13, color=FG_DARK, pad=20,
    )
    ax.grid(color=GRID_CLR, linewidth=0.8)
    ax.spines["polar"].set_color(GRID_CLR)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=10, framealpha=0.8, facecolor=BG_WHITE, edgecolor=GRID_CLR)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Radar chart saved → {save_path}")


if __name__ == "__main__":
    import sys
    # abspath avoids empty-string path when run from the module's own directory (Bug #13)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader       import load_config, load_iris_data
    from data_preprocessing import preprocess_pipeline
    from clustering_model  import train_kmeans, get_cluster_labels, get_cluster_centers
    from optimal_k_finder  import compute_elbow_curve, compute_silhouette_scores, compute_gap_statistic

    cfg = load_config()
    df  = load_iris_data(cfg, save_raw=False)
    X_scaled, feature_cols, scaler, df_clean = preprocess_pipeline(df, cfg)
    k_range = range(cfg["model"]["k_range_min"], cfg["model"]["k_range_max"] + 1)

    elbow_res    = compute_elbow_curve(X_scaled, k_range)
    sil_res      = compute_silhouette_scores(X_scaled, k_range)
    gap_res      = compute_gap_statistic(X_scaled, k_range)
    optimal_k    = 3

    km           = train_kmeans(X_scaled, k=optimal_k)
    labels       = get_cluster_labels(km)
    centers_orig = get_cluster_centers(km, scaler, feature_cols)

    plot_elbow_curve(elbow_res["k_values"], elbow_res["inertias"], optimal_k)
    plot_silhouette_scores(sil_res["k_values"], sil_res["silhouette_scores"],
                           X_scaled, labels, optimal_k)
    plot_gap_statistic(gap_res["k_values"], gap_res["gaps"], gap_res["sdk"], optimal_k)
    visualize_clusters_2d(X_scaled, labels, km.cluster_centers_, feature_cols, optimal_k)
    visualize_clusters_3d(X_scaled, labels, km.cluster_centers_, optimal_k)
    plot_feature_pairs(df_clean, labels, feature_cols, optimal_k)
    plot_radar_chart(centers_orig, feature_cols, optimal_k)
