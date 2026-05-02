#!/usr/bin/env python3
"""
K-Means MapReduce — Mapper

Reads data points from stdin (TSV: feature1\tfeature2\t...featureN).
Loads current centroids from 'centroids.txt' (distributed via Hadoop -files).
Assigns each point to nearest centroid, emits: cluster_id\tfeature1,feature2,...

Usage (Hadoop Streaming):
  hadoop jar hadoop-streaming.jar \
    -files centroids.txt \
    -mapper "python3 mapper.py" \
    -reducer "python3 reducer.py" \
    -input /input/data.tsv \
    -output /output/iteration_N
"""
import sys
import os
import math


def load_centroids(filepath="centroids.txt"):
    """Load centroids from file. Format: cluster_id\tf1,f2,...,fN"""
    centroids = {}
    # Look in current directory (Hadoop distributes file here)
    if not os.path.exists(filepath):
        # Fallback: try same directory as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filepath)

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            cluster_id = int(parts[0])
            coords = list(map(float, parts[1].split(",")))
            centroids[cluster_id] = coords
    return centroids


def euclidean_distance(point, centroid):
    """Compute Euclidean distance between two vectors."""
    return math.sqrt(sum((p - c) ** 2 for p, c in zip(point, centroid)))


def find_nearest_centroid(point, centroids):
    """Return the cluster_id of the nearest centroid."""
    min_dist = float("inf")
    nearest = -1
    for cluster_id, centroid in centroids.items():
        dist = euclidean_distance(point, centroid)
        if dist < min_dist:
            min_dist = dist
            nearest = cluster_id
    return nearest


def main():
    centroids = load_centroids()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            features = list(map(float, line.split("\t")))
        except ValueError:
            continue  # skip header or malformed lines

        nearest = find_nearest_centroid(features, centroids)

        # Emit: cluster_id \t feature1,feature2,...
        print(f"{nearest}\t{','.join(map(str, features))}")


if __name__ == "__main__":
    main()
