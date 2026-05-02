"""Quick validation of the comparison notebook logic."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import permutations
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
df = pd.read_csv("../../data/airbnb-cleaned.csv", low_memory=False)
features = ["Latitude", "Longitude", "Price", "Accommodates",
            "Bedrooms", "Bathrooms", "Review Scores Rating"]
X = df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Dataset: {X_scaled.shape[0]:,} rows x {X_scaled.shape[1]} features")

# sklearn K-Means
sklearn_km = KMeans(n_clusters=4, n_init=10, random_state=42)
sklearn_labels = sklearn_km.fit_predict(X_scaled)
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), size=30000, replace=False)
sklearn_sil = silhouette_score(X_scaled[sample_idx], sklearn_labels[sample_idx])
print(f"\nsklearn: Inertia={sklearn_km.inertia_:,.0f}, Silhouette={sklearn_sil:.4f}, Iters={sklearn_km.n_iter_}")

# MapReduce centroids
mr_centroids = {}
with open("centroids.txt", "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        cid = int(parts[0])
        coords = list(map(float, parts[1].split(",")))
        mr_centroids[cid] = coords
mr_centroids_array = np.array([mr_centroids[i] for i in range(4)])

# Assign labels
distances = np.linalg.norm(X_scaled[:, np.newaxis] - mr_centroids_array[np.newaxis, :], axis=2)
mr_labels = np.argmin(distances, axis=1)
mr_sil = silhouette_score(X_scaled[sample_idx], mr_labels[sample_idx])
mr_inertia = sum(np.sum((X_scaled[mr_labels == k] - mr_centroids_array[k]) ** 2) for k in range(4))
print(f"MapReduce: Inertia={mr_inertia:,.0f}, Silhouette={mr_sil:.4f}, Iters=20")

# Overlap
best_overlap = 0
best_perm = None
for perm in permutations(range(4)):
    overlap = sum(np.sum((sklearn_labels == i) & (mr_labels == perm[i])) for i in range(4))
    if overlap > best_overlap:
        best_overlap = overlap
        best_perm = perm

overlap_pct = 100 * best_overlap / len(sklearn_labels)
print(f"\nCluster mapping (sklearn→MR): {dict(enumerate(best_perm))}")
print(f"Assignment overlap: {best_overlap:,} / {len(sklearn_labels):,} ({overlap_pct:.1f}%)")

# Centroids in original scale
scaler_params = pd.read_csv("scaler_params.tsv", sep="\t")
means = scaler_params["mean"].values
stds = scaler_params["std"].values
sklearn_orig = sklearn_km.cluster_centers_ * stds + means
mr_orig = mr_centroids_array * stds + means

print(f"\n{'='*70}")
print(f"sklearn centroids (original scale):")
print(f"  {'Cluster':<8} {'Lat':>7} {'Lon':>8} {'Price':>7} {'Accom':>6} {'Beds':>5} {'Bath':>5} {'Rating':>7}")
for i in range(4):
    c = sklearn_orig[i]
    print(f"  {i:<8} {c[0]:>7.1f} {c[1]:>8.1f} {c[2]:>7.0f} {c[3]:>6.1f} {c[4]:>5.1f} {c[5]:>5.1f} {c[6]:>7.1f}")

print(f"\nMapReduce centroids (original scale):")
print(f"  {'Cluster':<8} {'Lat':>7} {'Lon':>8} {'Price':>7} {'Accom':>6} {'Beds':>5} {'Bath':>5} {'Rating':>7}")
for i in range(4):
    c = mr_orig[i]
    print(f"  {i:<8} {c[0]:>7.1f} {c[1]:>8.1f} {c[2]:>7.0f} {c[3]:>6.1f} {c[4]:>5.1f} {c[5]:>5.1f} {c[6]:>7.1f}")
print(f"{'='*70}")
