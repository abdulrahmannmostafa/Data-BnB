#!/usr/bin/env python3
"""
K-Means MapReduce — Data Preparation

Exports the clustering features from the cleaned dataset to a plain TSV file
suitable for Hadoop HDFS input.

Output format: Latitude\tLongitude\tPrice\tAccommodates\tBedrooms\tBathrooms\tReviewScoresRating
(No header — raw numeric values only, tab-separated)

Usage:
  python prepare_data.py [--sample 30000] [--output data.tsv]
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description="Prepare data for MapReduce K-Means")
    parser.add_argument("--input", default="../../data/airbnb-cleaned.csv",
                        help="Path to cleaned CSV")
    parser.add_argument("--output", default="data.tsv",
                        help="Output TSV file path")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N rows (None = use all)")
    parser.add_argument("--standardize", action="store_true", default=True,
                        help="Standardize features (recommended for K-Means)")
    parser.add_argument("--no-standardize", dest="standardize", action="store_false")
    args = parser.parse_args()

    # Resolve relative paths from script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} rows")

    # Select clustering features (same as sklearn version)
    features = ["Latitude", "Longitude", "Price", "Accommodates",
                "Bedrooms", "Bathrooms", "Review Scores Rating"]
    X = df[features].copy()

    # Sample if requested
    if args.sample:
        np.random.seed(42)
        X = X.sample(n=min(args.sample, len(X)), random_state=42)
        print(f"  Sampled {len(X):,} rows")

    # Standardize
    if args.standardize:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)
        X = X_scaled
        print("  Features standardized (zero mean, unit variance)")

        # Save scaler params for inverse transform later
        scaler_path = os.path.join(script_dir, "scaler_params.tsv")
        scaler_df = pd.DataFrame({
            "feature": features,
            "mean": scaler.mean_,
            "std": scaler.scale_
        })
        scaler_df.to_csv(scaler_path, sep="\t", index=False)
        print(f"  Scaler params saved to: {scaler_path}")

    # Save as TSV (no header, no index)
    X.to_csv(output_path, sep="\t", header=False, index=False)
    print(f"\nData exported to: {output_path}")
    print(f"  Shape: {X.shape}")
    print(f"  Format: TSV (tab-separated, no header)")
    print(f"\nFeature order: {', '.join(features)}")


if __name__ == "__main__":
    main()
