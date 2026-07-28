"""
optimal_k_finder.py
--------------------
OptimalKFinderModule: Implements three complementary methods to determine
the optimal number of clusters for K-Means:
  1. Elbow Method     (Inertia vs. k)
  2. Silhouette Score (Average silhouette for each k)
  3. Gap Statistic    (Compare inertia to random baseline)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ─────────────────────────────────────────────────────────────────────────────
#  Elbow Method
# ─────────────────────────────────────────────────────────────────────────────

def compute_elbow_curve(
    X: np.ndarray,
    k_range: range,
    random_seed: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> dict:
    """
    Compute inertia (within-cluster sum of squares) for each k in k_range.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    k_range : range
        Range of k values to evaluate (e.g. range(2, 11)).
    random_seed : int
    n_init : int
        Number of initializations per k.
    max_iter : int
    tol : float

    Returns
    -------
    dict with keys 'k_values' and 'inertias'.
    """
    inertias = []
    k_values = list(k_range)

    print("\n[OptimalK] Computing Elbow Curve (Inertia):")
    print(f"  {'k':>4} | {'Inertia':>12}")
    print("  " + "-" * 20)

    for k in k_values:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_seed,
        )
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"  {k:>4} | {km.inertia_:>12.4f}")

    return {"k_values": k_values, "inertias": inertias}


def find_elbow_point(inertias: list, k_values: list) -> int:
    """
    Identify the elbow point using the second-derivative (curvature) method.

    After two rounds of np.diff, second_diff[i] represents the curvature
    centered on inertias[i+2] (i.e. k_values[i+2]).  We therefore add 2
    (not 1) to map the argmax back to the correct k index.
    """
    if len(inertias) < 3:
        return k_values[0]

    inertias_arr = np.array(inertias, dtype=float)
    # Normalise to [0, 1] so all features are on equal footing
    spread = inertias_arr.max() - inertias_arr.min()
    normed = (inertias_arr - inertias_arr.min()) / (spread if spread > 0 else 1.0)
    # Second differences give a discrete approximation of curvature
    second_diff = np.diff(np.diff(normed))
    # second_diff[i]  <->  inertias[i+2]  <->  k_values[i+2]
    elbow_idx = int(np.argmax(second_diff)) + 2
    # Guard against out-of-range (edge case when argmax falls at last element)
    elbow_idx = min(elbow_idx, len(k_values) - 1)
    return k_values[elbow_idx]


# ─────────────────────────────────────────────────────────────────────────────
#  Silhouette Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_silhouette_scores(
    X: np.ndarray,
    k_range: range,
    random_seed: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> dict:
    """
    Compute the average silhouette score for each k in k_range.

    Returns
    -------
    dict with keys 'k_values' and 'silhouette_scores'.
    """
    scores = []
    k_values = list(k_range)

    print("\n[OptimalK] Computing Silhouette Scores:")
    print(f"  {'k':>4} | {'Silhouette Score':>18}")
    print("  " + "-" * 26)

    for k in k_values:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_seed,
        )
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
        print(f"  {k:>4} | {score:>18.4f}")

    best_k = k_values[int(np.argmax(scores))]
    print(f"\n  → Best k by Silhouette: {best_k} (score = {max(scores):.4f})")
    return {"k_values": k_values, "silhouette_scores": scores}


# ─────────────────────────────────────────────────────────────────────────────
#  Gap Statistic
# ─────────────────────────────────────────────────────────────────────────────

def _compute_inertia(X: np.ndarray, k: int, seed: int, n_init: int, max_iter: int, tol: float) -> float:
    km = KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                max_iter=max_iter, tol=tol, random_state=seed)
    km.fit(X)
    return km.inertia_


def compute_gap_statistic(
    X: np.ndarray,
    k_range: range,
    n_references: int = 10,
    random_seed: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> dict:
    """
    Compute the Gap Statistic for each k in k_range.

    The gap statistic compares the log-inertia of the actual clustering
    to the expected log-inertia of uniformly distributed reference data.

    Returns
    -------
    dict with keys 'k_values', 'gaps', 'sdk', and 'optimal_k'.
    """
    rng    = np.random.RandomState(random_seed)
    k_vals = list(k_range)
    gaps   = []
    sks    = []

    # Column-wise bounds for uniform reference data
    col_mins = X.min(axis=0)
    col_maxs = X.max(axis=0)

    print("\n[OptimalK] Computing Gap Statistic:")
    print(f"  {'k':>4} | {'Gap':>10} | {'s_k':>10}")
    print("  " + "-" * 30)

    for k in k_vals:
        actual_log_inertia = np.log(_compute_inertia(X, k, random_seed, n_init, max_iter, tol))

        ref_log_inertias = []
        for _ in range(n_references):
            X_ref = rng.uniform(col_mins, col_maxs, size=X.shape)
            ref_log_inertias.append(
                np.log(_compute_inertia(X_ref, k, random_seed, n_init, max_iter, tol))
            )

        ref_arr = np.array(ref_log_inertias)
        gap = ref_arr.mean() - actual_log_inertia
        sdk = np.sqrt(1 + 1 / n_references) * ref_arr.std()
        gaps.append(gap)
        sks.append(sdk)
        print(f"  {k:>4} | {gap:>10.4f} | {sdk:>10.4f}")

    # Optimal k: smallest k where gap(k) >= gap(k+1) - s_k+1
    optimal_k = k_vals[0]
    for i in range(len(k_vals) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            optimal_k = k_vals[i]
            break
    else:
        optimal_k = k_vals[np.argmax(gaps)]

    print(f"\n  → Optimal k by Gap Statistic: {optimal_k} (gap = {gaps[k_vals.index(optimal_k)]:.4f})")
    return {"k_values": k_vals, "gaps": gaps, "sdk": sks, "optimal_k": optimal_k}


# ─────────────────────────────────────────────────────────────────────────────
#  Aggregate Decision
# ─────────────────────────────────────────────────────────────────────────────

def determine_optimal_k(
    elbow_results: dict,
    silhouette_results: dict,
    gap_results: dict,
) -> int:
    """
    Combine evidence from all three methods to select the final optimal k.

    Priority:
      1. Gap statistic
      2. Silhouette score (highest)
      3. Elbow point (curvature)
    """
    k_gap  = gap_results["optimal_k"]
    k_sil  = silhouette_results["k_values"][
        int(np.argmax(silhouette_results["silhouette_scores"]))
    ]
    k_elb  = find_elbow_point(elbow_results["inertias"], elbow_results["k_values"])

    votes = [k_gap, k_sil, k_elb]
    vote_counts = {k: votes.count(k) for k in set(votes)}
    max_votes = max(vote_counts.values())
    # True majority (>1 vote) wins; otherwise gap statistic is the tie-breaker
    majority_winners = [k for k, v in vote_counts.items() if v == max_votes]
    if len(majority_winners) == 1:
        optimal_k  = majority_winners[0]
        decision   = f"k = {optimal_k} (majority vote, {max_votes}/3)"
    else:
        optimal_k  = k_gap  # gap statistic is the primary authority
        decision   = f"k = {optimal_k} (no majority — gap statistic tie-break)"

    print("\n" + "=" * 50)
    print("  OPTIMAL k SELECTION SUMMARY")
    print("=" * 50)
    print(f"  Elbow Method     : k = {k_elb}")
    print(f"  Silhouette Score : k = {k_sil}")
    print(f"  Gap Statistic    : k = {k_gap}")
    print(f"  Final Decision   : {decision}")
    print("=" * 50 + "\n")

    return optimal_k


if __name__ == "__main__":
    import sys, os
    # abspath avoids empty-string path when run from the module's own directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_config, load_iris_data
    from data_preprocessing import preprocess_pipeline

    cfg = load_config()
    df  = load_iris_data(cfg, save_raw=False)
    X_scaled, feature_cols, scaler, _ = preprocess_pipeline(df, cfg)

    k_range = range(cfg["model"]["k_range_min"], cfg["model"]["k_range_max"] + 1)
    seed    = cfg["model"]["random_seed"]

    elbow_res     = compute_elbow_curve(X_scaled, k_range, seed)
    silhouette_res = compute_silhouette_scores(X_scaled, k_range, seed)
    gap_res        = compute_gap_statistic(X_scaled, k_range,
                                           n_references=cfg["gap_statistic"]["n_references"],
                                           random_seed=seed)

    optimal_k = determine_optimal_k(elbow_res, silhouette_res, gap_res)
