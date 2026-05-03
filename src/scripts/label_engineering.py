import pandas as pd
import numpy as np

INPUT_PATH = "../../data/airbnb-cleaned.csv"
OUTPUT_PATH = "../../data/airbnb-labeled.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Shape: {df.shape}")

print("\nEngineering demand label...")

rev_per_month = df["Reviews per Month"].fillna(0)

num_reviews_max = df["Number of Reviews"].replace(0, np.nan).max()
num_reviews_norm = df["Number of Reviews"].fillna(0) / num_reviews_max

df["demand_score"] = 0.6 * rev_per_month + 0.4 * num_reviews_norm

median_score = df["demand_score"].median()
df["demand_label"] = (df["demand_score"] > median_score).astype(int)

print(f"\ndemand_score stats:\n{df['demand_score'].describe()}")
print(f"\nBinary label distribution:\n{df['demand_label'].value_counts()}")
print(
    f"Class balance: {df['demand_label'].value_counts(normalize=True).round(3).to_dict()}"
)

q33, q67 = df["demand_score"].quantile([1 / 3, 2 / 3]).values
print(f"\nTertile thresholds: 33rd pct = {q33:.4f}, 67th pct = {q67:.4f}")

df["demand_label_3"] = pd.qcut(df["demand_score"], q=3, labels=[0, 1, 2]).astype(int)

print(
    f"\n3-class label distribution:\n{df['demand_label_3'].value_counts().sort_index()}"
)

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved labeled dataset to: {OUTPUT_PATH}")
print(f"Final shape: {df.shape}")
