#!/usr/bin/env python3
"""
K-Means MapReduce — Reducer

Reads mapper output from stdin (sorted by key):
  cluster_id\tfeature1,feature2,...

Computes new centroid for each cluster (mean of all assigned points).
Emits: cluster_id\tnew_f1,new_f2,...,new_fN

Usage (Hadoop Streaming):
  (Used automatically as reducer in the streaming job)
"""
import sys


def main():
    current_cluster = None
    point_sum = None
    point_count = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            continue

        cluster_id = int(parts[0])
        features = list(map(float, parts[1].split(",")))

        if current_cluster == cluster_id:
            # Accumulate
            for i in range(len(features)):
                point_sum[i] += features[i]
            point_count += 1
        else:
            # Emit previous cluster centroid
            if current_cluster is not None and point_count > 0:
                centroid = [s / point_count for s in point_sum]
                print(f"{current_cluster}\t{','.join(map(str, centroid))}")

            # Start new cluster
            current_cluster = cluster_id
            point_sum = features[:]
            point_count = 1

    # Emit last cluster
    if current_cluster is not None and point_count > 0:
        centroid = [s / point_count for s in point_sum]
        print(f"{current_cluster}\t{','.join(map(str, centroid))}")


if __name__ == "__main__":
    main()
