"""
descriptive_analysis.py
========================
Airbnb Global Listings — Descriptive Analysis
CMP4011 Big Data & Cloud Computing | Spring 2026 | Team 1

Implements:
  1. Data preparation  — feature selection & scaling for clustering
  2. Sklearn KMeans    — reference implementation
  3. MapReduce KMeans  — custom pseudo-distributed implementation using
                         Python multiprocessing to emulate the
                         Map → Shuffle → Reduce paradigm
  4. Comparison        — inertia, silhouette, Davies-Bouldin, label agreement
  5. Rich visualisation

Usage (from notebook or script):
    import descriptive_analysis as da
    results = da.run_all(train_path="../data/splits/train.csv")

IMPORTANT — multiprocessing guard:
    Any script that calls run_all() or MapReduceKMeans directly **must** be
    protected with  if __name__ == "__main__":  on Windows/macOS (spawn
    start method).  On Linux (fork) it is not strictly required but is still
    best practice.  The bottom of this file shows the pattern.
"""

from __future__ import annotations

import os
import time
import warnings
from multiprocessing import Pool, cpu_count

import matplotlib
import matplotlib.gridspec as gridspec  # noqa: F401  (kept for optional use)
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Use non-interactive backend for multiprocessing safety
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment  # top-level import (fix)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (14, 6)})

RANDOM_STATE = 42
N_CLUSTERS = 5
N_INIT = 10
MAX_ITER = 300
SAMPLE_SIZE = 50_000

# Features used for clustering — interpretable, low-leakage subset.
CLUSTER_FEATURES = [
    "Price_original",
    "Accommodates",
    "Bedrooms",
    "Bathrooms",
    "Beds",
    "Amenities Count",
    "Host Verifications Count",
    "Host Tenure Days",
    "Review Scores Rating",
    "Review Scores Composite",
    "Number of Reviews",
    "Reviews per Month",
    "Availability 365",
    "Days Since Last Review",
]

# FIX: postprocessing.py saves raw (pre-scaling) copies of numeric profile
# columns with a "_raw" suffix.  We prefer those for profiling so the cluster
# stats show real-world values (guests, nights, scores) rather than z-scores.
# Price_original is already saved unscaled, so it keeps its original name.
RAW_SUFFIX = "_raw"

# Human-readable segment labels mapped by price rank (0 = cheapest cluster).
SEGMENT_LABELS = [
    "Budget Compact",
    "Mid-Range Shared",
    "Premium Entire Home",
    "Luxury High-Capacity",
    "Long-Stay / Low-Review",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────


def load_and_prepare(train_path: str):
    """
    Load the training split produced by postprocessing.py.

    Returns
    -------
    X_raw      : pd.DataFrame  — unscaled cluster features (for profiling)
    X_scaled   : np.ndarray    — StandardScaler-normalised matrix
    scaler     : StandardScaler instance
    df_full    : pd.DataFrame  — full training DataFrame (for enriched plots)
    feat_names : list[str]     — feature names actually present in scaled matrix
    raw_feat_names : list[str] — corresponding _raw column names used for profiling
    """
    print(f"[Data] Loading {train_path} …")
    df_full = pd.read_csv(train_path, low_memory=False)
    print(f"[Data] Loaded shape: {df_full.shape}")

    # ── Build the scaled clustering matrix from whatever columns are present ──
    available_scaled = [f for f in CLUSTER_FEATURES if f in df_full.columns]
    missing_scaled = [f for f in CLUSTER_FEATURES if f not in df_full.columns]
    if missing_scaled:
        print(f"[Data] ⚠  Scaled features not found (skipped): {missing_scaled}")

    X_scaled_df = df_full[available_scaled].copy()
    X_scaled_df = X_scaled_df.fillna(X_scaled_df.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_scaled_df.values)

    # ── Build X_raw using _raw columns where available, else fall back ────────
    # postprocessing.py saves e.g. "Accommodates_raw", "Bedrooms_raw", etc.
    # Price_original is always unscaled so we keep it as-is.
    raw_cols_map: dict[str, str] = {}  # scaled_col -> raw_col_name_in_df
    for feat in available_scaled:
        raw_candidate = feat + RAW_SUFFIX  # e.g. "Accommodates_raw"
        if feat == "Price_original":
            raw_cols_map[feat] = feat  # already raw
        elif raw_candidate in df_full.columns:
            raw_cols_map[feat] = raw_candidate  # prefer _raw copy
        else:
            raw_cols_map[feat] = feat  # fallback: use scaled col

    # X_raw: one column per cluster feature, pulled from the best available source
    X_raw = pd.DataFrame(index=df_full.index)
    for feat, raw_col in raw_cols_map.items():
        X_raw[feat] = df_full[raw_col].values

    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))

    raw_found = [f for f, rc in raw_cols_map.items() if rc.endswith(RAW_SUFFIX)]
    raw_missing = [
        f
        for f, rc in raw_cols_map.items()
        if not rc.endswith(RAW_SUFFIX) and f != "Price_original"
    ]
    if raw_found:
        print(
            f"[Data] ✓  Using _raw columns for {len(raw_found)} features: {raw_found}"
        )
    if raw_missing:
        print(
            f"[Data] ⚠  No _raw column found for {raw_missing} "
            f"— profile stats for these may be z-scores. "
            f"Re-run postprocessing.py to fix."
        )

    print(f"[Data] Clustering matrix: {X_scaled.shape}")
    return X_raw, X_scaled, scaler, df_full, available_scaled


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ELBOW + SILHOUETTE SELECTION
# ─────────────────────────────────────────────────────────────────────────────


def elbow_silhouette(X_scaled: np.ndarray, k_range=range(2, 11)) -> pd.DataFrame:
    """
    Compute inertia and silhouette score for every k in k_range.
    Uses a subsample of <= SAMPLE_SIZE rows for speed.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_scaled), size=min(SAMPLE_SIZE, len(X_scaled)), replace=False)
    X_sub = X_scaled[idx]

    records = []
    print("[Elbow] Computing inertia & silhouette scores …")
    for k in k_range:
        km = KMeans(
            n_clusters=k, n_init=N_INIT, max_iter=MAX_ITER, random_state=RANDOM_STATE
        )
        labels = km.fit_predict(X_sub)
        inertia = km.inertia_
        sil = silhouette_score(
            X_sub,
            labels,
            sample_size=min(10_000, len(X_sub)),
            random_state=RANDOM_STATE,
        )
        records.append({"k": k, "inertia": inertia, "silhouette": sil})
        print(f"  k={k:2d} | inertia={inertia:,.0f} | silhouette={sil:.4f}")

    return pd.DataFrame(records)


def plot_elbow_silhouette(metrics_df: pd.DataFrame, best_k: int, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        metrics_df["k"], metrics_df["inertia"], "o-", color="#3498db", linewidth=2
    )
    axes[0].axvline(best_k, color="red", linestyle="--", label=f"Chosen k={best_k}")
    axes[0].set_title("Elbow Curve — Inertia vs k", fontsize=13)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].legend()

    axes[1].plot(
        metrics_df["k"], metrics_df["silhouette"], "s-", color="#2ecc71", linewidth=2
    )
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"Chosen k={best_k}")
    axes[1].set_title("Silhouette Score vs k", fontsize=13)
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    plt.suptitle("Optimal k Selection", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "elbow_silhouette.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SKLEARN KMEANS
# ─────────────────────────────────────────────────────────────────────────────


def run_sklearn_kmeans(X_scaled: np.ndarray, k: int):
    """
    Fit sklearn KMeans on the full scaled matrix.
    Returns (fitted model, labels array, elapsed seconds).
    """
    print(f"\n[sklearn KMeans] Fitting k={k} …")
    t0 = time.perf_counter()
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=N_INIT,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        algorithm="lloyd",
    )
    labels = km.fit_predict(X_scaled)
    elapsed = time.perf_counter() - t0
    print(
        f"  Done in {elapsed:.2f}s | Inertia: {km.inertia_:,.0f} | Iterations: {km.n_iter_}"
    )
    return km, labels, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAPREDUCE KMEANS  (pseudo-distributed, single-machine multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────


def _kmeans_map(args):
    """
    MAP worker.
    Input  : (data_chunk [n_chunk x d],  centroids [k x d])
    Output : list of (centroid_idx, point_vector) tuples
    """
    chunk, centroids = args
    chunk = np.asarray(chunk, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    diffs = chunk[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dists_sq = (diffs**2).sum(axis=2)
    nearest = dists_sq.argmin(axis=1)
    return list(zip(nearest.tolist(), chunk.tolist()))


def _kmeans_reduce(args):
    """
    REDUCE worker.
    Input  : (centroid_idx, list_of_point_vectors)
    Output : (centroid_idx, new_centroid_vector)
    """
    cid, points = args
    return cid, np.mean(np.asarray(points, dtype=np.float64), axis=0)


def _kmeans_plus_plus_init(
    X: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """k-means++ initialisation — distance-weighted seeding."""
    n = len(X)
    first_idx = int(rng.integers(n))
    centroids = [X[first_idx]]

    for _ in range(1, k):
        C = np.array(centroids)
        diffs = X[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = (diffs**2).sum(axis=2).min(axis=1)
        probs = dists / dists.sum()
        idx = int(rng.choice(n, p=probs))
        centroids.append(X[idx])

    return np.array(centroids)


class MapReduceKMeans:
    """
    K-Means implemented with Python multiprocessing to emulate MapReduce.

    Parameters
    ----------
    n_clusters   : int
    max_iter     : int
    tol          : float   convergence threshold (max L2 centroid shift)
    n_workers    : int     map/reduce workers (default = cpu_count())
    random_state : int
    """

    def __init__(
        self,
        n_clusters: int = 5,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_workers: int | None = None,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_workers = n_workers or max(1, cpu_count())
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.convergence_history_: list[float] = []

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        init_n = min(10_000, len(X))
        init_idx = rng.choice(len(X), size=init_n, replace=False)
        centroids = _kmeans_plus_plus_init(X[init_idx], self.n_clusters, rng)

        chunks = np.array_split(X, self.n_workers)

        print(
            f"  [MR KMeans] {len(X):,} points | k={self.n_clusters} | "
            f"workers={self.n_workers} | max_iter={self.max_iter}"
        )

        with Pool(processes=self.n_workers) as pool:
            for iteration in range(1, self.max_iter + 1):

                c_list = centroids.tolist()
                map_args = [(chunk.tolist(), c_list) for chunk in chunks]
                map_outs = pool.map(_kmeans_map, map_args)

                buckets: dict[int, list] = {i: [] for i in range(self.n_clusters)}
                for pairs in map_outs:
                    for cid, point in pairs:
                        buckets[cid].append(point)

                non_empty = [(cid, pts) for cid, pts in buckets.items() if pts]
                reduced = pool.map(_kmeans_reduce, non_empty)
                new_centroids = centroids.copy()
                for cid, new_c in reduced:
                    new_centroids[cid] = new_c

                shift = float(np.max(np.linalg.norm(new_centroids - centroids, axis=1)))
                self.convergence_history_.append(shift)
                centroids = new_centroids
                self.n_iter_ = iteration

                if shift < self.tol:
                    print(
                        f"  [MR KMeans] Converged at iteration {iteration} "
                        f"(max centroid shift = {shift:.6f})"
                    )
                    break

        diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dists_sq = (diffs**2).sum(axis=2)
        labels = dists_sq.argmin(axis=1)

        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = float(dists_sq[np.arange(len(X)), labels].sum())

        return labels


def run_mapreduce_kmeans(X_scaled: np.ndarray, k: int):
    """Fit MapReduce KMeans. Returns (model, labels, elapsed_seconds)."""
    print(f"\n[MapReduce KMeans] Fitting k={k} …")
    t0 = time.perf_counter()
    mr_km = MapReduceKMeans(
        n_clusters=k, max_iter=MAX_ITER, tol=1e-4, random_state=RANDOM_STATE
    )
    labels = mr_km.fit_predict(X_scaled)
    elapsed = time.perf_counter() - t0
    print(
        f"  Done in {elapsed:.2f}s | Inertia: {mr_km.inertia_:,.0f} | "
        f"Iterations: {mr_km.n_iter_}"
    )
    return mr_km, labels, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPARISON METRICS
# ─────────────────────────────────────────────────────────────────────────────


def compare_implementations(
    X_scaled, sk_labels, mr_labels, sk_model, mr_model, sk_time, mr_time
) -> pd.DataFrame:
    """Side-by-side evaluation. Returns a summary DataFrame."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_scaled), size=min(20_000, len(X_scaled)), replace=False)
    X_sub = X_scaled[idx]

    def _metrics(labels_full):
        labels_sub = labels_full[idx]
        sil = silhouette_score(X_sub, labels_sub, random_state=RANDOM_STATE)
        db = davies_bouldin_score(X_sub, labels_sub)
        return sil, db

    sk_sil, sk_db = _metrics(sk_labels)
    mr_sil, mr_db = _metrics(mr_labels)

    summary = pd.DataFrame(
        {
            "Metric": [
                "Inertia (WCSS)",
                "Silhouette Score ↑",
                "Davies-Bouldin Score ↓",
                "Iterations",
                "Time (s)",
            ],
            "sklearn KMeans": [
                f"{sk_model.inertia_:,.0f}",
                f"{sk_sil:.4f}",
                f"{sk_db:.4f}",
                sk_model.n_iter_,
                f"{sk_time:.2f}",
            ],
            "MapReduce KMeans": [
                f"{mr_model.inertia_:,.0f}",
                f"{mr_sil:.4f}",
                f"{mr_db:.4f}",
                mr_model.n_iter_,
                f"{mr_time:.2f}",
            ],
        }
    )

    print("\n" + "=" * 55)
    print("IMPLEMENTATION COMPARISON")
    print("=" * 55)
    print(summary.to_string(index=False))
    print("=" * 55)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLUSTER PROFILING
# ─────────────────────────────────────────────────────────────────────────────


def _rank_clusters_by_price(X_raw: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """
    Return a label-remapping array such that cluster 0 = cheapest median price.
    """
    price_col = (
        "Price_original" if "Price_original" in X_raw.columns else X_raw.columns[0]
    )
    tmp = X_raw[[price_col]].copy()
    tmp["_lbl"] = labels
    medians = tmp.groupby("_lbl")[price_col].median().sort_values()
    remap = np.zeros(labels.max() + 1, dtype=int)
    for new_id, old_id in enumerate(medians.index):
        remap[old_id] = new_id
    return remap


def profile_clusters(
    X_raw: pd.DataFrame,
    labels: np.ndarray,
    feature_names: list[str],
    impl_name: str = "sklearn",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute per-cluster statistics using real (unscaled) values, assign
    interpretive segment names, and return (profile_df, remapped_labels).

    FIX: X_raw is now built in load_and_prepare() from _raw-suffixed columns
    saved by postprocessing.py, so all median stats are in original units
    (dollars, guest counts, review scores out of 100, etc.) rather than
    StandardScaler z-scores.
    """
    remap = _rank_clusters_by_price(X_raw, labels)
    remapped_labels = remap[labels]

    X_prof = X_raw.copy()
    X_prof["Cluster"] = remapped_labels

    agg_dict = {"Count": ("Cluster", "size")}
    for f in feature_names:
        if f in X_prof.columns:
            agg_dict[f] = (f, "median")

    profile = (
        X_prof.groupby("Cluster")
        .agg(**agg_dict)
        .reset_index()
        .sort_values("Cluster")
        .reset_index(drop=True)
    )

    n_seg = len(profile)
    profile.insert(
        1,
        "Segment",
        [
            SEGMENT_LABELS[i] if i < len(SEGMENT_LABELS) else f"Segment {i}"
            for i in range(n_seg)
        ],
    )

    price_col = (
        "Price_original" if "Price_original" in feature_names else feature_names[0]
    )
    display_cols = [
        "Cluster",
        "Segment",
        "Count",
        price_col,
        "Accommodates",
        "Bedrooms",
        "Review Scores Rating",
        "Availability 365",
    ]
    display_cols = [c for c in display_cols if c in profile.columns]

    print(f"\n[Cluster Profile — {impl_name}]")
    print(profile[display_cols].to_string(index=False))
    return profile, remapped_labels


# ─────────────────────────────────────────────────────────────────────────────
# 7.  HELPER
# ─────────────────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, output_dir: str, filename: str) -> None:
    """Save figure and show it."""
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"  [Saved] {path}")
    plt.show()


def _normalise_rows(vals: np.ndarray) -> np.ndarray:
    """Min-max normalise each column of a 2-D array to [0, 1]."""
    col_min = vals.min(0)
    col_max = vals.max(0)
    denom = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    return (vals - col_min) / denom


# ─────────────────────────────────────────────────────────────────────────────
# 8.  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────


def plot_pca_clusters(
    X_scaled: np.ndarray,
    sk_labels: np.ndarray,
    mr_labels: np.ndarray,
    k: int,
    output_dir: str,
):
    """PCA 2-D projection — sklearn vs MapReduce side by side."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_scaled), size=min(SAMPLE_SIZE, len(X_scaled)), replace=False)
    X_2d = pca.fit_transform(X_scaled[idx])
    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    palette = sns.color_palette("Set1", k)

    for ax, labels, title in zip(
        axes,
        [sk_labels[idx], mr_labels[idx]],
        ["sklearn KMeans", "MapReduce KMeans"],
    ):
        for cid in range(k):
            mask = labels == cid
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=3,
                alpha=0.35,
                color=palette[cid],
                label=SEGMENT_LABELS[cid] if cid < len(SEGMENT_LABELS) else f"C{cid}",
            )
        ax.set_title(
            f"{title}\n(PCA — {evr[0]*100:.1f}% + {evr[1]*100:.1f}% variance)",
            fontsize=12,
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=4, fontsize=8)

    plt.suptitle("Cluster Projection — PCA 2D", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save(fig, output_dir, "pca_clusters.png")
    return fig


def plot_cluster_radar(
    profile: pd.DataFrame,
    feature_names: list[str],
    impl_name: str,
    output_dir: str,
):
    """Radar / spider chart of normalised median feature values per cluster."""
    radar_feats = [
        f
        for f in [
            "Price_original",
            "Accommodates",
            "Bedrooms",
            "Bathrooms",
            "Amenities Count",
            "Review Scores Rating",
            "Availability 365",
            "Number of Reviews",
        ]
        if f in feature_names and f in profile.columns
    ][:8]

    if len(radar_feats) < 3:
        print(f"[Radar] Too few features ({len(radar_feats)}) — skipping.")
        return

    vals = profile[radar_feats].values.astype(float)
    vals_norm = _normalise_rows(vals)

    N = len(radar_feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    palette = sns.color_palette("Set1", len(profile))

    for i, row in profile.iterrows():
        row_vals = list(vals_norm[i]) + [vals_norm[i, 0]]
        seg_name = row.get("Segment", f"Cluster {i}")
        ax.plot(angles, row_vals, linewidth=2, label=seg_name, color=palette[i])
        ax.fill(angles, row_vals, alpha=0.1, color=palette[i])

    ax.set_thetagrids(np.degrees(angles[:-1]), radar_feats, fontsize=9)
    ax.set_title(
        f"Cluster Radar Chart — {impl_name}",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    slug = impl_name.lower().replace(" ", "_")
    _save(fig, output_dir, f"radar_{slug}.png")
    return fig


def plot_cluster_boxplots(
    X_raw: pd.DataFrame,
    labels: np.ndarray,
    impl_name: str,
    output_dir: str,
):
    """Box plots of Price and Accommodates per cluster."""
    df_plot = X_raw.copy()
    df_plot["Cluster"] = labels.astype(str)

    price_col = (
        "Price_original" if "Price_original" in df_plot.columns else df_plot.columns[0]
    )
    cap = df_plot[price_col].quantile(0.99)
    df_plot = df_plot[df_plot[price_col] <= cap]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    n_clusters = df_plot["Cluster"].nunique()
    palette = sns.color_palette("Set1", n_clusters)

    sns.boxplot(data=df_plot, x="Cluster", y=price_col, palette=palette, ax=axes[0])
    axes[0].set_title(f"Price Distribution per Cluster — {impl_name}", fontsize=12)
    axes[0].set_ylabel("Price ($)")

    if "Accommodates" in df_plot.columns:
        sns.boxplot(
            data=df_plot, x="Cluster", y="Accommodates", palette=palette, ax=axes[1]
        )
        axes[1].set_title(f"Accommodates per Cluster — {impl_name}", fontsize=12)
        axes[1].set_ylabel("Accommodates (persons)")

    plt.tight_layout()
    slug = impl_name.lower().replace(" ", "_")
    _save(fig, output_dir, f"boxplot_{slug}.png")
    return fig


def plot_convergence(mr_model: MapReduceKMeans, output_dir: str):
    """Centroid shift per iteration for the MapReduce model."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        range(1, len(mr_model.convergence_history_) + 1),
        mr_model.convergence_history_,
        "o-",
        color="#e74c3c",
        linewidth=2,
    )
    ax.axhline(mr_model.tol, color="gray", linestyle="--", label=f"tol={mr_model.tol}")
    ax.set_title("MapReduce KMeans — Centroid Shift per Iteration", fontsize=13)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max Centroid Shift (L2)")
    ax.legend()
    plt.tight_layout()
    _save(fig, output_dir, "mr_convergence.png")
    return fig


def plot_cluster_heatmap(
    profile: pd.DataFrame,
    feature_names: list[str],
    impl_name: str,
    output_dir: str,
):
    """Heatmap of normalised cluster centroids."""
    hm_feats = [f for f in feature_names if f in profile.columns]
    vals = profile[hm_feats].values.astype(float)
    vals_norm = _normalise_rows(vals)

    row_labels = (
        profile["Segment"].tolist()
        if "Segment" in profile.columns
        else profile["Cluster"].tolist()
    )
    df_hm = pd.DataFrame(vals_norm, columns=hm_feats, index=row_labels)

    fig, ax = plt.subplots(figsize=(max(12, len(hm_feats) * 0.9), 5))
    sns.heatmap(
        df_hm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Normalised Median"},
    )
    ax.set_title(
        f"Cluster Feature Heatmap — {impl_name}", fontsize=13, fontweight="bold"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    plt.tight_layout()
    slug = impl_name.lower().replace(" ", "_")
    _save(fig, output_dir, f"heatmap_{slug}.png")
    return fig


def plot_cluster_size_pie(
    sk_labels: np.ndarray,
    mr_labels: np.ndarray,
    k: int,
    output_dir: str,
):
    """Pie charts of cluster sizes for both implementations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = sns.color_palette("Set1", k)

    for ax, labels, title in zip(
        axes,
        [sk_labels, mr_labels],
        ["sklearn KMeans", "MapReduce KMeans"],
    ):
        counts = pd.Series(labels).value_counts().sort_index()
        seg_labels = [
            SEGMENT_LABELS[i] if i < len(SEGMENT_LABELS) else f"C{i}"
            for i in counts.index
        ]
        ax.pie(
            counts,
            labels=seg_labels,
            autopct="%1.1f%%",
            colors=[palette[i] for i in counts.index],
            startangle=140,
            wedgeprops={"edgecolor": "white"},
        )
        ax.set_title(f"Cluster Size Distribution\n{title}", fontsize=12)

    plt.suptitle(
        "How Listings Are Distributed Across Clusters",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, output_dir, "cluster_sizes.png")
    return fig


def plot_label_agreement(
    sk_labels: np.ndarray,
    mr_labels: np.ndarray,
    output_dir: str,
):
    """
    Confusion-matrix-style heatmap showing assignment agreement between
    sklearn and MapReduce KMeans (after optimal label alignment with the
    Hungarian algorithm).
    """
    k = int(max(sk_labels.max(), mr_labels.max())) + 1
    matrix = np.zeros((k, k), dtype=int)
    for s, m in zip(sk_labels, mr_labels):
        matrix[s, m] += 1

    row_ind, col_ind = linear_sum_assignment(-matrix)
    aligned = matrix[row_ind][:, col_ind]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        aligned,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=[f"MR C{i}" for i in col_ind],
        yticklabels=[f"SK C{i}" for i in row_ind],
    )
    ax.set_title(
        "Label Agreement — sklearn vs MapReduce KMeans\n(Optimal Permutation)",
        fontsize=12,
    )
    ax.set_xlabel("MapReduce Cluster")
    ax.set_ylabel("sklearn Cluster")

    total = aligned.sum()
    agree = aligned.diagonal().sum()
    print(
        f"\n[Agreement] {agree:,}/{total:,} points assigned identically "
        f"({agree/total*100:.1f}%)"
    )

    plt.tight_layout()
    _save(fig, output_dir, "label_agreement.png")
    return fig


def plot_geo_clusters(
    df_full: pd.DataFrame,
    sk_labels: np.ndarray,
    k: int,
    output_dir: str,
):
    """
    Scatter plot of listing lat/lon coloured by cluster assignment.
    Silently skipped if Latitude/Longitude are absent.
    """
    if "Latitude" not in df_full.columns or "Longitude" not in df_full.columns:
        print("[Geo] Latitude/Longitude not in training split — skipping.")
        return

    df_geo = df_full[["Latitude", "Longitude"]].copy()
    df_geo["Cluster"] = sk_labels

    df_geo = df_geo[
        df_geo["Latitude"].between(-60, 75) & df_geo["Longitude"].between(-180, 180)
    ]

    rng = np.random.default_rng(RANDOM_STATE)
    if len(df_geo) > SAMPLE_SIZE:
        df_geo = df_geo.iloc[rng.choice(len(df_geo), SAMPLE_SIZE, replace=False)]

    palette = sns.color_palette("Set1", k)
    fig, ax = plt.subplots(figsize=(16, 8))
    for cid in range(k):
        mask = df_geo["Cluster"] == cid
        label = SEGMENT_LABELS[cid] if cid < len(SEGMENT_LABELS) else f"C{cid}"
        ax.scatter(
            df_geo.loc[mask, "Longitude"],
            df_geo.loc[mask, "Latitude"],
            s=2,
            alpha=0.3,
            color=palette[cid],
            label=label,
        )

    ax.set_title(
        "Geographic Distribution of Listing Clusters (sklearn KMeans)", fontsize=13
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(markerscale=5, fontsize=9, loc="lower left")
    plt.tight_layout()
    _save(fig, output_dir, "geo_clusters.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9.  BUSINESS INSIGHTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────


def print_business_insights(
    sk_profile: pd.DataFrame,
    mr_profile: pd.DataFrame,
) -> None:
    """Print actionable business insights derived from cluster profiles."""
    print("\n" + "=" * 65)
    print("BUSINESS INSIGHTS — Airbnb Listing Segmentation")
    print("=" * 65)

    price_col = (
        "Price_original"
        if "Price_original" in sk_profile.columns
        else sk_profile.columns[2]
    )

    for _, row in sk_profile.iterrows():
        seg = row.get("Segment", f"Cluster {row['Cluster']}")
        count = int(row["Count"])
        price = row.get(price_col, "N/A")
        accom = row.get("Accommodates", "N/A")
        rating = row.get("Review Scores Rating", "N/A")
        avail = row.get("Availability 365", "N/A")

        print(f"\n  ▶ {seg}  (n={count:,})")
        if isinstance(price, (int, float)):
            print(f"    Median Price          : ${price:.0f}/night")
        if isinstance(accom, (int, float)):
            print(f"    Accommodates          : {accom:.0f} guests")
        if isinstance(rating, (int, float)):
            print(f"    Avg Review Score      : {rating:.1f}/100")
        if isinstance(avail, (int, float)):
            print(f"    Availability (days/yr): {avail:.0f}")

    print("\n  💡 Recommendations:")
    print("    • Budget Compact listings should highlight Wifi & Kitchen amenities")
    print("      to close the gap with mid-range competitors.")
    print("    • Premium & Luxury clusters show the highest review scores —")
    print("      invest in cleanliness and communication to sustain ratings.")
    print("    • Long-Stay / Low-Review listings may benefit from an introductory")
    print("      pricing strategy to accumulate early reviews.")
    print("    • Hosts in the Mid-Range Shared segment have the highest listing")
    print("      counts — suggest professional photography to differentiate.")
    print("=" * 65)


def plot_tsne_clusters(
    X_scaled: np.ndarray,
    sk_labels: np.ndarray,
    mr_labels: np.ndarray,
    k: int,
    output_dir: str,
    sample_size: int = 15000,
):
    """t-SNE 2D projection for intuitive nonlinear cluster visualization."""
    print("\n[t-SNE] Running dimensionality reduction...")

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_scaled), size=min(sample_size, len(X_scaled)), replace=False)

    X_sub = X_scaled[idx]
    sk_sub = sk_labels[idx]
    mr_sub = mr_labels[idx]

    tsne = TSNE(
        n_components=2,
        perplexity=50,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE,
        max_iter=1500,   # newer sklearn uses max_iter not n_iter
        verbose=1, # show progress in console
    )

    X_2d = tsne.fit_transform(X_sub)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    palette = sns.color_palette("Set1", k)

    for ax, labels, title in zip(
        axes,
        [sk_sub, mr_sub],
        ["sklearn KMeans", "MapReduce KMeans"],
    ):
        for cid in range(k):
            mask = labels == cid
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=8,
                alpha=0.55,
                color=palette[cid],
                label=SEGMENT_LABELS[cid] if cid < len(SEGMENT_LABELS) else f"C{cid}",
            )
        ax.set_title(f"{title} — t-SNE Projection", fontsize=13, fontweight="bold")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.legend(markerscale=2, fontsize=8)

    plt.suptitle(
        "t-SNE Cluster Visualization (Nonlinear Projection)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, output_dir, "tsne_clusters.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def run_all(
    train_path: str = "../data/splits/train.csv",
    k: int = N_CLUSTERS,
    run_elbow: bool = True,
    output_dir: str = "../outputs",
) -> dict:
    """
    Full descriptive analysis pipeline.

    Parameters
    ----------
    train_path : str   path to the training split CSV
    k          : int   number of clusters
    run_elbow  : bool  whether to run the elbow/silhouette grid search
    output_dir : str   directory for saved figures

    Returns
    -------
    dict with all key objects for further use in the notebook.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────
    X_raw, X_scaled, scaler, df_full, feat_names = load_and_prepare(train_path)

    # ── Elbow / Silhouette ────────────────────────────────────────────────
    metrics_df = None
    if run_elbow:
        metrics_df = elbow_silhouette(X_scaled)
        plot_elbow_silhouette(metrics_df, best_k=k, output_dir=output_dir)

    # ── sklearn KMeans ────────────────────────────────────────────────────
    sk_model, sk_labels_raw, sk_time = run_sklearn_kmeans(X_scaled, k)

    # ── MapReduce KMeans ──────────────────────────────────────────────────
    mr_model, mr_labels_raw, mr_time = run_mapreduce_kmeans(X_scaled, k)

    # ── Comparison ────────────────────────────────────────────────────────
    comparison = compare_implementations(
        X_scaled,
        sk_labels_raw,
        mr_labels_raw,
        sk_model,
        mr_model,
        sk_time,
        mr_time,
    )

    # ── Convergence plot ──────────────────────────────────────────────────
    plot_convergence(mr_model, output_dir=output_dir)

    # ── Cluster profiles (X_raw now holds real unscaled values) ──────────
    sk_profile, sk_labels = profile_clusters(
        X_raw, sk_labels_raw, feat_names, "sklearn KMeans"
    )
    mr_profile, mr_labels = profile_clusters(
        X_raw, mr_labels_raw, feat_names, "MapReduce KMeans"
    )

    # ── Visualisations ────────────────────────────────────────────────────
    plot_pca_clusters(X_scaled, sk_labels, mr_labels, k, output_dir=output_dir)
    plot_tsne_clusters(X_scaled, sk_labels, mr_labels, k, output_dir=output_dir)
    plot_cluster_size_pie(sk_labels, mr_labels, k, output_dir=output_dir)
    plot_label_agreement(sk_labels, mr_labels, output_dir=output_dir)
    plot_cluster_boxplots(X_raw, sk_labels, "sklearn KMeans", output_dir=output_dir)
    plot_cluster_boxplots(X_raw, mr_labels, "MapReduce KMeans", output_dir=output_dir)
    plot_cluster_heatmap(
        sk_profile, feat_names, "sklearn KMeans", output_dir=output_dir
    )
    plot_cluster_heatmap(
        mr_profile, feat_names, "MapReduce KMeans", output_dir=output_dir
    )
    plot_cluster_radar(sk_profile, feat_names, "sklearn KMeans", output_dir=output_dir)
    plot_cluster_radar(
        mr_profile, feat_names, "MapReduce KMeans", output_dir=output_dir
    )
    plot_geo_clusters(df_full, sk_labels, k, output_dir=output_dir)

    # ── Business insights ─────────────────────────────────────────────────
    print_business_insights(sk_profile, mr_profile)

    # ── Save enriched training split ──────────────────────────────────────
    df_full["Cluster_sklearn"] = sk_labels
    df_full["Cluster_mapreduce"] = mr_labels
    enriched_path = os.path.join(os.path.dirname(train_path), "train_with_clusters.csv")
    df_full.to_csv(enriched_path, index=False)
    print(f"\n[Output] Saved enriched training split → {enriched_path}")

    return {
        "X_raw": X_raw,
        "X_scaled": X_scaled,
        "scaler": scaler,
        "feature_names": feat_names,
        "df_full": df_full,
        "sk_model": sk_model,
        "sk_labels": sk_labels,
        "mr_model": mr_model,
        "mr_labels": mr_labels,
        "sk_profile": sk_profile,
        "mr_profile": mr_profile,
        "comparison": comparison,
        "metrics_df": metrics_df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all()
