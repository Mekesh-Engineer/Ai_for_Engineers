"""
clustering_model.py
--------------------
ClusteringModule: Train the final K-Means model with the optimal k,
extract cluster labels and centers, and persist the trained model.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score  # moved to module level (Bug #9)


def train_kmeans(
    X: np.ndarray,
    k: int,
    random_seed: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> KMeans:
    """
    Train a K-Means model with k-means++ initialization.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    k : int
        Number of clusters.
    random_seed : int
    n_init : int
    max_iter : int
    tol : float

    Returns
    -------
    Fitted KMeans instance.
    """
    print(f"\n[Clustering] Training K-Means with k = {k} …")
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_seed,
    )
    km.fit(X)
    print(f"  Iterations to convergence : {km.n_iter_}")
    print(f"  Final inertia             : {km.inertia_:.4f}")
    # Convert numpy integer types to plain Python ints for clean printing (Bug #7)
    sizes = {int(label): int(count)
             for label, count in zip(*np.unique(km.labels_, return_counts=True))}
    print(f"  Cluster sizes             : {sizes}")
    return km


def get_cluster_labels(km: KMeans) -> np.ndarray:
    """Return the cluster assignment (label) for every sample."""
    return km.labels_


def get_cluster_centers(
    km: KMeans,
    scaler,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Return cluster centers in the *original* (un-scaled) feature space.

    Parameters
    ----------
    km : KMeans
        Fitted K-Means model.
    scaler : fitted scaler
        Used to inverse-transform the scaled centers.
    feature_cols : list
        Column names corresponding to features.

    Returns
    -------
    pd.DataFrame with one row per cluster and one column per feature.
    """
    centers_scaled   = km.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
    df_centers = pd.DataFrame(centers_original, columns=feature_cols)
    df_centers.index.name = "cluster"
    return df_centers


def save_model(km: KMeans, model_path: str) -> None:
    """Persist the trained K-Means model to disk as a pickle file."""
    # Use abspath so dirname is never empty for bare filenames (Bug #8)
    model_dir = os.path.dirname(os.path.abspath(model_path))
    os.makedirs(model_dir, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(km, f)
    print(f"[Clustering] Model saved → {model_path}")


def load_model(model_path: str) -> KMeans:
    """Load a persisted K-Means model from disk."""
    with open(model_path, "rb") as f:
        km = pickle.load(f)
    print(f"[Clustering] Model loaded ← {model_path}")
    return km


def run_stability_check(
    X: np.ndarray,
    k: int,
    n_runs: int = 5,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_init: int = 10,
) -> dict:
    """
    Assess clustering stability by running K-Means with different seeds
    and reporting variation in inertia and silhouette scores.

    Returns
    -------
    dict with 'inertias', 'seeds', 'inertia_std'.
    """
    seeds   = [42, 7, 123, 999, 2024][:n_runs]
    inertias = []
    sils     = []

    print(f"\n[Clustering] Stability check across {n_runs} seeds:")
    print(f"  {'Seed':>6} | {'Inertia':>12} | {'Silhouette':>12}")
    print("  " + "-" * 36)

    for seed in seeds:
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                    max_iter=max_iter, tol=tol, random_state=seed)
        km.fit(X)
        sil = silhouette_score(X, km.labels_)
        inertias.append(km.inertia_)
        sils.append(sil)
        print(f"  {seed:>6} | {km.inertia_:>12.4f} | {sil:>12.4f}")

    print(f"\n  Inertia std  : {np.std(inertias):.4f}")
    print(f"  Silhouette std : {np.std(sils):.6f}")
    return {"seeds": seeds, "inertias": inertias, "silhouettes": sils,
            "inertia_std": np.std(inertias)}


if __name__ == "__main__":
    import sys, os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_config, load_iris_data
    from data_preprocessing import preprocess_pipeline

    cfg = load_config()
    df  = load_iris_data(cfg, save_raw=False)
    X_scaled, feature_cols, scaler, _ = preprocess_pipeline(df, cfg)

    optimal_k = 3  # known optimal for Iris
    km = train_kmeans(X_scaled, k=optimal_k,
                      random_seed=cfg["model"]["random_seed"],
                      n_init=cfg["model"]["n_init"],
                      max_iter=cfg["model"]["max_iter"],
                      tol=cfg["model"]["tol"])

    labels  = get_cluster_labels(km)
    centers = get_cluster_centers(km, scaler, feature_cols)
    print("\nCluster centers (original scale):\n", centers.round(3))

    save_model(km, cfg["output"]["model_path"])
    run_stability_check(X_scaled, optimal_k)
