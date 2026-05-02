#!/usr/bin/env python3
"""
K-Means MapReduce — Local Driver

Simulates Hadoop MapReduce locally by piping data through mapper | sort | reducer.
Iterates until convergence or max iterations reached.

This lets you test the MapReduce K-Means before deploying to Hadoop.
On Hadoop, the driver logic is replaced by a shell script or Oozie workflow
that re-runs the streaming job each iteration with updated centroids.

Usage:
  python driver.py [--k 4] [--max-iter 20] [--tol 1e-4] [--input data.tsv]
"""
import os
import sys
import math
import random
import argparse
import subprocess


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPER = os.path.join(SCRIPT_DIR, "mapper.py")
REDUCER = os.path.join(SCRIPT_DIR, "reducer.py")
CENTROIDS_FILE = os.path.join(SCRIPT_DIR, "centroids.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="K-Means MapReduce Local Driver")
    parser.add_argument("--input", default=os.path.join(SCRIPT_DIR, "data_full.tsv"),
                        help="Path to input TSV data file")
    parser.add_argument("--k", type=int, default=4, help="Number of clusters")
    parser.add_argument("--max-iter", type=int, default=20, help="Max iterations")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="Convergence tolerance (max centroid shift)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_data_sample(filepath, n=None):
    """Load data points from TSV file (optionally just first n lines)."""
    points = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                point = list(map(float, line.split("\t")))
                points.append(point)
            except ValueError:
                continue  # skip header
            if n and len(points) >= n:
                break
    return points


def initialize_centroids(data_path, k, seed=42):
    """Random initialization: pick k random points from data as initial centroids."""
    points = load_data_sample(data_path)
    random.seed(seed)
    initial = random.sample(points, k)
    return {i: c for i, c in enumerate(initial)}


def write_centroids(centroids, filepath):
    """Write centroids to file. Format: cluster_id\\tf1,f2,..."""
    with open(filepath, "w") as f:
        for cid in sorted(centroids.keys()):
            coords = ",".join(map(str, centroids[cid]))
            f.write(f"{cid}\t{coords}\n")


def read_centroids(filepath):
    """Read centroids from reducer output file."""
    centroids = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            cid = int(parts[0])
            coords = list(map(float, parts[1].split(",")))
            centroids[cid] = coords
    return centroids


def centroid_shift(old_centroids, new_centroids):
    """Compute maximum Euclidean shift across all centroids."""
    max_shift = 0.0
    for cid in old_centroids:
        if cid in new_centroids:
            dist = math.sqrt(sum(
                (a - b) ** 2
                for a, b in zip(old_centroids[cid], new_centroids[cid])
            ))
            max_shift = max(max_shift, dist)
    return max_shift


def run_mapreduce_iteration(data_path, python_exe):
    """
    Simulate one MapReduce iteration locally:
      cat data.tsv | python mapper.py | sort | python reducer.py
    """
    # Build the pipeline command
    if sys.platform == "win32":
        cmd = (
            f'type "{data_path}" | '
            f'"{python_exe}" "{MAPPER}" | '
            f'sort | '
            f'"{python_exe}" "{REDUCER}"'
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    else:
        cmd = (
            f'cat "{data_path}" | '
            f'"{python_exe}" "{MAPPER}" | '
            f'sort -t$"\\t" -k1,1n | '
            f'"{python_exe}" "{REDUCER}"'
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR in MapReduce iteration:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    return result.stdout


def main():
    args = parse_args()
    python_exe = sys.executable

    print(f"K-Means MapReduce (Local Simulation)")
    print(f"  Input: {args.input}")
    print(f"  K={args.k}, max_iter={args.max_iter}, tol={args.tol}")
    print()

    # Step 1: Initialize centroids
    centroids = initialize_centroids(args.input, args.k, args.seed)
    write_centroids(centroids, CENTROIDS_FILE)
    print(f"Initialized {args.k} centroids from random data points.")

    # Step 2: Iterate
    for iteration in range(1, args.max_iter + 1):
        # Write current centroids for mapper to read
        write_centroids(centroids, CENTROIDS_FILE)

        # Run one MapReduce iteration
        output = run_mapreduce_iteration(args.input, python_exe)

        # Parse new centroids from reducer output
        new_centroids = {}
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            cid = int(parts[0])
            coords = list(map(float, parts[1].split(",")))
            new_centroids[cid] = coords

        # Compute convergence
        shift = centroid_shift(centroids, new_centroids)
        centroids = new_centroids

        print(f"  Iteration {iteration:2d} | max centroid shift = {shift:.6f}")

        if shift < args.tol:
            print(f"\n  Converged after {iteration} iterations (shift < {args.tol})")
            break
    else:
        print(f"\n  Reached max iterations ({args.max_iter})")

    # Save final centroids
    write_centroids(centroids, CENTROIDS_FILE)
    print(f"\nFinal centroids saved to: {CENTROIDS_FILE}")

    # Print final centroids
    print("\nFinal Centroids:")
    feature_names = ["Latitude", "Longitude", "Price", "Accommodates",
                     "Bedrooms", "Bathrooms", "Review Scores Rating"]
    print(f"  {'Cluster':<8}", end="")
    for name in feature_names:
        print(f"{name:>12}", end="")
    print()
    for cid in sorted(centroids.keys()):
        print(f"  {cid:<8}", end="")
        for val in centroids[cid]:
            print(f"{val:>12.2f}", end="")
        print()


if __name__ == "__main__":
    main()
