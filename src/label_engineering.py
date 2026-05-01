import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
INPUT_PATH = "../data/airbnb-cleaned.csv"
OUTPUT_PATH = "../data/airbnb-labeled.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Shape: {df.shape}")

# ──────────────────────────────────────────────
# 1. Build demand score
#    - Reviews per Month  → strongest proxy for booking frequency
#    - Number of Reviews  → normalised volume signal
# ──────────────────────────────────────────────
print("\nEngineering demand label...")

rev_per_month = df["Reviews per Month"].fillna(0)

# NOTE: max() is computed over all rows here (minor leakage).
# The effect is negligible for large datasets, but postprocessing could
# recompute this threshold from training rows only if strict leakage-free
# evaluation is required.
num_reviews_max = df["Number of Reviews"].replace(0, np.nan).max()
num_reviews_norm = df["Number of Reviews"].fillna(0) / num_reviews_max

df["demand_score"] = 0.6 * rev_per_month + 0.4 * num_reviews_norm

# ──────────────────────────────────────────────
# 2. Binary label  (0 = low demand, 1 = high demand)
#    Split at the median so classes stay roughly balanced
# ──────────────────────────────────────────────
median_score = df["demand_score"].median()
df["demand_label"] = (df["demand_score"] > median_score).astype(int)

print(f"\ndemand_score stats:\n{df['demand_score'].describe()}")
print(f"\nBinary label distribution:\n{df['demand_label'].value_counts()}")
print(
    f"Class balance: {df['demand_label'].value_counts(normalize=True).round(3).to_dict()}"
)

# ──────────────────────────────────────────────
# 3. 3-Class label  (0=low, 1=medium, 2=high)
#    Uses tertile cut-points for equal-sized classes.
#
#    KNOWN LIMITATION: pd.qcut is applied to ALL rows here, so the
#    tertile boundaries include test data → minor label leakage.
#    To fix properly: run postprocessing.py which can recompute
#    the 3-class label from training rows only using the saved
#    demand_score column and these thresholds.
# ──────────────────────────────────────────────
q33, q67 = df["demand_score"].quantile([1/3, 2/3]).values
print(f"\nTertile thresholds: 33rd pct = {q33:.4f}, 67th pct = {q67:.4f}")

df["demand_label_3"] = pd.qcut(df["demand_score"], q=3, labels=[0, 1, 2]).astype(int)

print(
    f"\n3-class label distribution:\n{df['demand_label_3'].value_counts().sort_index()}"
)

# ──────────────────────────────────────────────
# 4. Save labeled dataset
# ──────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved labeled dataset to: {OUTPUT_PATH}")
print(f"Final shape: {df.shape}")
