# -*- coding: utf-8 -*-
"""
main.py
--------
End-to-end pipeline runner for the K-Means Clustering Experiment.
Orchestrates all modules: data loading → preprocessing → optimal-k
discovery → model training → evaluation → visualisation → report.

Usage
-----
    python main.py
    python main.py --config config/parameters.json
    python main.py --k 3          # force a specific k
"""

import argparse
import os
import sys
import time

# Ensure src/ is importable when running from project root
# Use abspath so sys.path is correct regardless of the working directory
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from data_loader      import load_config, load_iris_data, explore_dataset, verify_data_structure
from data_preprocessing import preprocess_pipeline
from optimal_k_finder  import (
    compute_elbow_curve,
    compute_silhouette_scores,
    compute_gap_statistic,
    determine_optimal_k,
)
from clustering_model  import (
    train_kmeans,
    get_cluster_labels,
    get_cluster_centers,
    save_model,
    run_stability_check,
)
from evaluation        import (
    evaluate_clustering,
    analyze_cluster_characteristics,
    generate_clustering_report,
    save_cluster_labels,
    save_cluster_centers,
)
from visualization     import (
    plot_elbow_curve,
    plot_silhouette_scores,
    plot_gap_statistic,
    visualize_clusters_2d,
    visualize_clusters_3d,
    plot_feature_pairs,
    plot_radar_chart,
)


# ─────────────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    banner = (
        "\n"
        "  +==============================================================+\n"
        "  |         K-MEANS CLUSTERING  -  Experiment 3                  |\n"
        "  |         Dataset : Iris (150 samples x 4 features)            |\n"
        "  +==============================================================+\n"
    )
    print(banner)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K-Means Clustering Pipeline — Experiment 3"
    )
    # Default is relative to the script location, not the CWD
    _default_cfg = os.path.join(_PROJECT_ROOT, "config", "parameters.json")
    parser.add_argument(
        "--config", default=_default_cfg,
        help="Path to the parameters.json config file.",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Force a specific k instead of auto-detecting the optimal one.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    args    = parse_args()
    print_banner()

    # ── 1. Load configuration ──────────────────────────────────────────────
    print("STEP 1 ▶ Load configuration")
    cfg = load_config(args.config)
    seed    = cfg["model"]["random_seed"]
    n_init  = cfg["model"]["n_init"]
    max_iter= cfg["model"]["max_iter"]
    tol     = cfg["model"]["tol"]
    k_range = range(cfg["model"]["k_range_min"], cfg["model"]["k_range_max"] + 1)

    # ── 2. Data collection & EDA ───────────────────────────────────────────
    print("\nSTEP 2 ▶ Data collection & exploration")
    df = load_iris_data(cfg, save_raw=True)
    explore_dataset(df)
    assert verify_data_structure(df), "Data validation failed — check your dataset."

    # ── 3. Preprocessing ───────────────────────────────────────────────────
    print("\nSTEP 3 ▶ Preprocessing (imputation → outlier treatment → scaling)")
    X_scaled, feature_cols, scaler, df_clean = preprocess_pipeline(df, cfg)

    # ── 4. Optimal-k discovery ─────────────────────────────────────────────
    print("\nSTEP 4 ▶ Optimal-k discovery")
    elbow_res = compute_elbow_curve(X_scaled, k_range, seed, n_init, max_iter, tol)
    sil_res   = compute_silhouette_scores(X_scaled, k_range, seed, n_init, max_iter, tol)
    gap_res   = compute_gap_statistic(
        X_scaled, k_range,
        n_references=cfg["gap_statistic"]["n_references"],
        random_seed=seed, n_init=n_init, max_iter=max_iter, tol=tol,
    )

    if args.k is not None:
        optimal_k = args.k
        k_selection_rationale = f"User-forced k={optimal_k}"
        print(f"\n  [Override] Using user-specified k = {optimal_k}")
    else:
        optimal_k = determine_optimal_k(elbow_res, sil_res, gap_res)
        # Build an accurate rationale that reflects what each method voted
        from optimal_k_finder import find_elbow_point
        import numpy as np
        _k_elb = find_elbow_point(elbow_res["inertias"], elbow_res["k_values"])
        _k_sil = sil_res["k_values"][int(np.argmax(sil_res["silhouette_scores"]))]
        _k_gap = gap_res["optimal_k"]
        _votes = [_k_elb, _k_sil, _k_gap]
        k_selection_rationale = (
            f"Majority vote — Elbow: k={_k_elb}, Silhouette: k={_k_sil}, "
            f"Gap Statistic: k={_k_gap}. Final: k={optimal_k}"
        )

    # ── 5. Model training ──────────────────────────────────────────────────
    print("\nSTEP 5 ▶ Model training")
    km = train_kmeans(X_scaled, k=optimal_k, random_seed=seed,
                      n_init=n_init, max_iter=max_iter, tol=tol)
    labels       = get_cluster_labels(km)
    centers_orig = get_cluster_centers(km, scaler, feature_cols)

    # ── 6. Stability check ─────────────────────────────────────────────────
    print("\nSTEP 6 ▶ Stability analysis")
    stability = run_stability_check(X_scaled, optimal_k, n_runs=5,
                                    max_iter=max_iter, tol=tol, n_init=n_init)

    # ── 7. Evaluation ──────────────────────────────────────────────────────
    print("\nSTEP 7 ▶ Evaluation")
    metrics         = evaluate_clustering(X_scaled, labels, km.inertia_)
    cluster_summary = analyze_cluster_characteristics(df_clean, labels, feature_cols, centers_orig)

    # ── 8. Save artefacts ──────────────────────────────────────────────────
    print("\nSTEP 8 ▶ Saving artefacts")
    save_model(km, cfg["output"]["model_path"])
    save_cluster_labels(labels, cfg["output"]["cluster_labels_csv"])
    save_cluster_centers(centers_orig, cfg["output"]["cluster_centers_csv"])
    generate_clustering_report(
        metrics, cluster_summary, centers_orig, stability,
        optimal_k, k_selection_rationale,
        cfg["output"]["report_txt"],
    )

    # ── 9. Visualisations ──────────────────────────────────────────────────
    print("\nSTEP 9 ▶ Generating visualisations")
    plot_elbow_curve(
        elbow_res["k_values"], elbow_res["inertias"],
        optimal_k, cfg["output"]["elbow_plot"],
    )
    plot_silhouette_scores(
        sil_res["k_values"], sil_res["silhouette_scores"],
        X_scaled, labels, optimal_k, cfg["output"]["silhouette_plot"],
    )
    plot_gap_statistic(
        gap_res["k_values"], gap_res["gaps"], gap_res["sdk"],
        optimal_k, "results/gap_statistic_plot.png",
    )
    visualize_clusters_2d(
        X_scaled, labels, km.cluster_centers_,
        feature_cols, optimal_k, cfg["output"]["cluster_2d_plot"],
    )
    visualize_clusters_3d(
        X_scaled, labels, km.cluster_centers_,
        optimal_k, cfg["output"]["cluster_3d_plot"],
    )
    plot_feature_pairs(
        df_clean, labels, feature_cols, optimal_k,
        "results/feature_pair_plot.png",
    )
    plot_radar_chart(
        centers_orig, feature_cols, optimal_k,
        "results/cluster_radar_chart.png",
    )

    # ── 10. Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Optimal k selected          : {optimal_k}")
    print(f"  Inertia                     : {metrics['inertia']}")
    print(f"  Silhouette Score            : {metrics['silhouette_score']}")
    print(f"  Davies-Bouldin Index        : {metrics['davies_bouldin_index']}")
    print(f"  Calinski-Harabasz Index     : {metrics['calinski_harabasz_index']}")
    print(f"  Total runtime               : {elapsed:.1f}s")
    print(f"\n  Outputs written to          : results/")
    print(f"  Model saved to              : {cfg['output']['model_path']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
