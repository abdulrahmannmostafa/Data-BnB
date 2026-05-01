import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import Counter

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
INPUT_PATH = "../data/airbnb-labeled.csv"  # output of label_engineering.py
OUTPUT_DIR = "../data/splits/"
ENCODER_DIR = "../encoders/"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)

# Columns to preserve in raw (unscaled) form for cluster profiling
RAW_PROFILE_COLS = [
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

print("Loading labeled dataset...")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Shape on load: {df.shape}")

# ──────────────────────────────────────────────
# FIX 1 — Deduplicate on ID
# ──────────────────────────────────────────────
print("\n[Fix 1] Deduplication...")
if "ID" in df.columns:
    before = len(df)
    df = df.drop_duplicates(subset=["ID"])
    print(f"  Removed {before - len(df)} duplicate rows.")
else:
    print("  'ID' not found — skipped.")

# ──────────────────────────────────────────────
# FIX 2 — Parse currency columns
# ──────────────────────────────────────────────
print("\n[Fix 2] Parsing currency columns...")
currency_cols = ["Price", "Cleaning Fee", "Security Deposit", "Extra People"]

for col in currency_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .str.strip()
            .replace(["", "nan", "None"], np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"  Parsed '{col}'.")

# ──────────────────────────────────────────────
# FIX 3 — Parse Host Response Rate
# ──────────────────────────────────────────────
print("\n[Fix 3] Parsing Host Response Rate...")
if "Host Response Rate" in df.columns:
    df["Host Response Rate"] = (
        df["Host Response Rate"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace(["", "nan", "None"], np.nan)
    )
    df["Host Response Rate"] = pd.to_numeric(df["Host Response Rate"], errors="coerce")

# ──────────────────────────────────────────────
# FIX 4 — Log-transform price (regression target)
# ──────────────────────────────────────────────
print("\n[Fix 4] Log-transforming price target...")
df = df[df["Price"].notna() & (df["Price"] > 0)]
df["Price_log"] = np.log1p(df["Price"])

# ──────────────────────────────────────────────
# FIX 5 — Review score composite
# ──────────────────────────────────────────────
print("\n[Fix 5] Review score composite...")
review_sub_cols = [
    "Review Scores Accuracy",
    "Review Scores Cleanliness",
    "Review Scores Checkin",
    "Review Scores Communication",
    "Review Scores Location",
    "Review Scores Value",
]
available_review_cols = [c for c in review_sub_cols if c in df.columns]
if available_review_cols:
    df["Review Scores Composite"] = df[available_review_cols].mean(axis=1)
    print(f"  Composite built from {len(available_review_cols)} sub-scores.")

# ──────────────────────────────────────────────
# FIX 6 — Train / Test split BEFORE any leaky transforms
#          FIX: stratify on demand_label for balanced splits
# ──────────────────────────────────────────────
print("\n[Fix 6] Stratified Train/Test split...")

REGRESSION_TARGET = "Price_log"
CLASSIFICATION_TARGET_BIN = "demand_label"  # binary
CLASSIFICATION_TARGET_MULTI = "demand_label_3"  # 3-class

DROP_FROM_FEATURES = [
    "Price",
    "Price_log",
    "demand_score",
    "demand_label",
    "demand_label_3",
]
if "ID" in df.columns:
    DROP_FROM_FEATURES.append("ID")

X = df.drop(columns=[c for c in DROP_FROM_FEATURES if c in df.columns])
y_reg = df[REGRESSION_TARGET]
y_cls = df[CLASSIFICATION_TARGET_BIN]
y_cls3 = df[CLASSIFICATION_TARGET_MULTI]

# FIX: stratify so class proportions are identical in train & test
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    random_state=42,
    stratify=y_cls,  # ← stratify on the classification label
)

y_cls_train = y_cls.loc[X_train.index]
y_cls_test = y_cls.loc[X_test.index]
y_cls3_train = y_cls3.loc[X_train.index]
y_cls3_test = y_cls3.loc[X_test.index]

print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"  Binary label - Train: {y_cls_train.value_counts().to_dict()}")
print(f"  Binary label - Test : {y_cls_test.value_counts().to_dict()}")

# ──────────────────────────────────────────────
# FIX 6b — Snapshot raw profile columns BEFORE scaling
# ──────────────────────────────────────────────
print("\n[Fix 6b] Snapshotting raw profile columns...")
raw_train_cols = {
    c: X_train[c].copy() for c in RAW_PROFILE_COLS if c in X_train.columns
}
raw_test_cols = {c: X_test[c].copy() for c in RAW_PROFILE_COLS if c in X_test.columns}
print(f"  Snapshotted {len(raw_train_cols)} columns.")

# ──────────────────────────────────────────────
# FIX 6c — Outlier Capping (Train-only percentiles)
#    As per EDA: Minimum Nights, Maximum Nights, Extra People,
#    Cleaning Fee, Security Deposit, and Price all have extreme outliers.
# ──────────────────────────────────────────────
print("\n[Fix 6c] Capping extreme outliers (1st-99th percentile of train)...")
outlier_cols = [
    "Minimum Nights",
    "Maximum Nights",
    "Extra People",
    "Cleaning Fee",
    "Security Deposit",
    "Host Listings Count"
]
for col in outlier_cols:
    if col in X_train.columns:
        q01 = X_train[col].quantile(0.01)
        q99 = X_train[col].quantile(0.99)
        X_train[col] = X_train[col].clip(lower=q01, upper=q99)
        X_test[col]  = X_test[col].clip(lower=q01, upper=q99)

# Cap the regression target (Price_log) based on train bounds
q01_y = y_reg_train.quantile(0.01)
q99_y = y_reg_train.quantile(0.99)
y_reg_train = y_reg_train.clip(lower=q01_y, upper=q99_y)
y_reg_test  = y_reg_test.clip(lower=q01_y, upper=q99_y)
print("  Outliers capped.")

# ──────────────────────────────────────────────
# FIX 7 — REMOVED: Price_vs_city_median
#    This feature divided each listing's Price by the city median Price.
#    Since Price IS the regression target, this constitutes target leakage
#    for the classification task (demand_label derived separately, but the
#    feature directly encodes the target variable's magnitude).
#    Amenities Count + City label encoding already provide city-level signal.
# ──────────────────────────────────────────────
print("\n[Fix 7] Price_vs_city_median SKIPPED (target leakage — see comment).")

# ──────────────────────────────────────────────
# FIX 8 — Encode categoricals AFTER split
# ──────────────────────────────────────────────
print("\n[Fix 8] Encoding categoricals...")

# FIX 5: Top-20 Amenities One-Hot Encoding
# As discovered in EDA, Top amenities appear in the vast majority of listings.
if "Parsed Amenities" in X_train.columns:
    print("\n  Extracting Top-20 Amenities...")
    amenity_counts = Counter()
    for row in X_train["Parsed Amenities"].dropna():
        amenity_counts.update(row.split("|"))
    
    top_20_amenities = [a[0] for a in amenity_counts.most_common(20) if a[0]]
    
    for amenity in top_20_amenities:
        col_name = f"Amenity_{amenity}"
        # We use regex=False because amenities might contain regex characters like parentheses
        X_train[col_name] = X_train["Parsed Amenities"].str.contains(amenity, regex=False, na=False).astype(int)
        X_test[col_name]  = X_test["Parsed Amenities"].str.contains(amenity, regex=False, na=False).astype(int)
    
    X_train.drop(columns=["Parsed Amenities"], inplace=True)
    X_test.drop(columns=["Parsed Amenities"],  inplace=True)
    print(f"  Created {len(top_20_amenities)} one-hot columns for top amenities.")

categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

for col in categorical_cols:
    unique_count = X_train[col].nunique(dropna=False)

    if unique_count > 15:
        le = LabelEncoder()
        X_train[col] = X_train[col].fillna("Unknown").astype(str)
        X_test[col] = X_test[col].fillna("Unknown").astype(str)

        le.fit(X_train[col])
        known = set(le.classes_)

        X_test[col] = X_test[col].apply(lambda x: x if x in known else "Unknown")

        if "Unknown" not in le.classes_:
            le.classes_ = np.append(le.classes_, "Unknown")

        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

        joblib.dump(le, os.path.join(ENCODER_DIR, f"{col}_label_encoder.pkl"))
    else:
        train_dummies = pd.get_dummies(
            X_train[col].fillna("Unknown"), prefix=col, drop_first=True
        )
        test_dummies = pd.get_dummies(
            X_test[col].fillna("Unknown"), prefix=col, drop_first=True
        )
        test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

        X_train = pd.concat([X_train.drop(columns=[col]), train_dummies], axis=1)
        X_test = pd.concat([X_test.drop(columns=[col]), test_dummies], axis=1)

joblib.dump(X_train.columns.tolist(), os.path.join(ENCODER_DIR, "feature_columns.pkl"))

# ──────────────────────────────────────────────
# FIX 9 — Train-only median imputation
# ──────────────────────────────────────────────
print("\n[Fix 9] Train-only median imputation...")
train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)
joblib.dump(train_medians, os.path.join(ENCODER_DIR, "train_medians.pkl"))

# ──────────────────────────────────────────────
# FIX 10 — VIF check
# ──────────────────────────────────────────────
print("\n[Fix 10] VIF check...")
vif_candidates = [
    c
    for c in [
        "Accommodates",
        "Bedrooms",
        "Bathrooms",
        "Beds",
        "Amenities Count",
        "Host Verifications Count",
    ]
    if c in X_train.columns
]
if len(vif_candidates) >= 2:
    vif_data = X_train[vif_candidates].dropna()
    vif_df = pd.DataFrame(
        {
            "Feature": vif_candidates,
            "VIF": [
                variance_inflation_factor(vif_data.values, i)
                for i in range(len(vif_candidates))
            ],
        }
    ).sort_values("VIF", ascending=False)
    print(vif_df.to_string(index=False))

# ──────────────────────────────────────────────
# FIX 11 — Standard scaling
# ──────────────────────────────────────────────
print("\n[Fix 11] Feature scaling...")
scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=[np.number]).columns

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

joblib.dump(scaler, os.path.join(ENCODER_DIR, "standard_scaler.pkl"))

# ──────────────────────────────────────────────
# FIX 12 — Save splits with raw columns + all targets re-attached
# ──────────────────────────────────────────────
print("\n[Fix 12] Saving splits...")

# Save original df-aligned indices BEFORE reset_index — needed for df.loc[] lookups below.
train_orig_idx = y_reg_train.index
test_orig_idx  = y_reg_test.index

# Reset index so that integer-based .values alignment is correct after
# pd.concat (get_dummies) operations that may have shuffled the positional index.
X_train = X_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)

train_df = X_train.copy()
train_df[REGRESSION_TARGET] = y_reg_train.values
train_df[CLASSIFICATION_TARGET_BIN] = y_cls_train.values
train_df[CLASSIFICATION_TARGET_MULTI] = y_cls3_train.values
train_df["Price_original"] = df.loc[train_orig_idx, "Price"].values

for col, series in raw_train_cols.items():
    train_df[f"{col}_raw"] = series.reset_index(drop=True).values

test_df = X_test.copy()
test_df[REGRESSION_TARGET] = y_reg_test.values
test_df[CLASSIFICATION_TARGET_BIN] = y_cls_test.values
test_df[CLASSIFICATION_TARGET_MULTI] = y_cls3_test.values
test_df["Price_original"] = df.loc[test_orig_idx, "Price"].values

for col, series in raw_test_cols.items():
    test_df[f"{col}_raw"] = series.reset_index(drop=True).values

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("POSTPROCESSING COMPLETE")
print("=" * 60)
print(f"Train rows              : {len(train_df):,}")
print(f"Test rows               : {len(test_df):,}")
print(f"Final feature count     : {X_train.shape[1]}")
print(f"Remaining train nulls   : {X_train.isna().sum().sum()}")
print(f"Saved artifacts         : {ENCODER_DIR}")
print(f"Saved splits            : {OUTPUT_DIR}")
print("=" * 60)
print("\nTargets saved in CSVs:")
print(f"  Regression  → {REGRESSION_TARGET}")
print(f"  Binary cls  → {CLASSIFICATION_TARGET_BIN}   (0=low, 1=high demand)")
print(f"  Multi  cls  → {CLASSIFICATION_TARGET_MULTI}  (0=low, 1=med, 2=high)")
print("\nUse np.expm1(predictions) to restore dollar prices from Price_log.")
