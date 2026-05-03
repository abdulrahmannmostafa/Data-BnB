import pandas as pd
import numpy as np
import joblib
import os
import re
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

INPUT_PATH = "../../data/airbnb-labeled.csv"
OUTPUT_DIR = "../../data/splits/"
ENCODER_DIR = "../../encoders/"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)

RAW_PROFILE_COLS = [
    "Accommodates",
    "Bedrooms",
    "Bathrooms",
    "Beds",
    "Amenities Count",
    "Host Verifications Count",
    "Host Tenure Days",
    "Review Scores Rating",
    "Number of Reviews",
    "Reviews per Month",
    "Availability 365",
    "Days Since Last Review",
]

REGRESSION_TARGET = "Price_log"
CLASSIFICATION_TARGET_BIN = "demand_label"
CLASSIFICATION_TARGET_MULTI = "demand_label_3"

print("Loading labeled dataset...")
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"Shape on load: {df.shape}")

# Guard: make sure label columns exist
for col in [CLASSIFICATION_TARGET_BIN, CLASSIFICATION_TARGET_MULTI, "demand_score"]:
    assert col in df.columns, f"Missing required column: '{col}'"


print("\n[Step 1] Deduplication...")
if "ID" in df.columns:
    before = len(df)
    df = df.drop_duplicates(subset=["ID"])
    print(f"  Removed {before - len(df)} duplicate rows.")
else:
    print("  'ID' column not found — skipped.")


print("\n[Step 2] Parsing currency columns...")
currency_cols = ["Price", "Cleaning Fee", "Security Deposit", "Extra People"]

for col in currency_cols:
    if col in df.columns and df[col].dtype == object:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .str.strip()
            .replace(["", "nan", "None"], np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"  Parsed '{col}'.")
    else:
        print(f"  '{col}' already numeric or not found — skipped.")

print("\n[Step 3] Parsing Host Response Rate...")
if "Host Response Rate" in df.columns and df["Host Response Rate"].dtype == object:
    df["Host Response Rate"] = (
        df["Host Response Rate"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace(["", "nan", "None"], np.nan)
    )
    df["Host Response Rate"] = pd.to_numeric(df["Host Response Rate"], errors="coerce")
    print("  Parsed.")
else:
    print("  Already numeric or not found — skipped.")


print("\n[Step 4] Filtering zero/null Price and log-transforming...")
before = len(df)
df = df[df["Price"].notna() & (df["Price"] > 0)].reset_index(drop=True)
print(f"  Removed {before - len(df)} rows with Price ≤ 0 or null.")
df["Price_log"] = np.log1p(df["Price"])
df["Price_original"] = df["Price"].copy()  # ← store now, before any split
print(f"  Price_log range: [{df['Price_log'].min():.3f}, {df['Price_log'].max():.3f}]")


print("\n[Step 5] Defining features and splitting...")

# Columns that must never appear as features
NON_FEATURE_COLS = [
    "Price",
    "Price_log",
    "Price_original",
    "demand_score",
    "demand_label",
    "demand_label_3",
    # leaky: used to construct demand_score / demand_label
    "Number of Reviews",
    "Reviews per Month",
    "Availability 30",
    "Availability 60",
    "Availability 90",
    "Availability 365",
]
if "ID" in df.columns:
    NON_FEATURE_COLS.append("ID")

feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
X = df[feature_cols].copy()
y_reg = df[REGRESSION_TARGET].copy()
y_cls = df[CLASSIFICATION_TARGET_BIN].copy()
y_cls3 = df[CLASSIFICATION_TARGET_MULTI].copy()

# Stratify on binary label so class proportions match in both splits
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42, stratify=y_cls
)

# Derive all other targets by the same index
y_cls_train = y_cls.loc[X_train.index]
y_cls_test = y_cls.loc[X_test.index]
y_cls3_train = y_cls3.loc[X_train.index]
y_cls3_test = y_cls3.loc[X_test.index]

print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"  Binary label — Train: {y_cls_train.value_counts().to_dict()}")
print(f"  Binary label — Test : {y_cls_test.value_counts().to_dict()}")
print(f"  3-class label — Train: {y_cls3_train.value_counts().sort_index().to_dict()}")
print(f"  3-class label — Test : {y_cls3_test.value_counts().sort_index().to_dict()}")


print("\n[Step 6] Snapshotting raw profile columns...")
raw_train = {c: X_train[c].copy() for c in RAW_PROFILE_COLS if c in X_train.columns}
raw_test = {c: X_test[c].copy() for c in RAW_PROFILE_COLS if c in X_test.columns}
print(f"  Snapshotted {len(raw_train)} columns.")


print("\n[Step 7] Capping outliers using train-only quantiles...")
outlier_cols = [
    "Minimum Nights",
    "Maximum Nights",
    "Extra People",
    "Cleaning Fee",
    "Security Deposit",
    "Host Listings Count",
]
clip_bounds = {}
for col in outlier_cols:
    if col in X_train.columns:
        q01 = X_train[col].quantile(0.01)
        q99 = X_train[col].quantile(0.99)
        clip_bounds[col] = (q01, q99)
        X_train[col] = X_train[col].clip(q01, q99)
        X_test[col] = X_test[col].clip(q01, q99)

# Cap regression target
q01_y = y_reg_train.quantile(0.01)
q99_y = y_reg_train.quantile(0.99)
clip_bounds["Price_log"] = (q01_y, q99_y)
y_reg_train = y_reg_train.clip(q01_y, q99_y)
y_reg_test = y_reg_test.clip(q01_y, q99_y)

# Save ALL clip bounds for inference
joblib.dump(clip_bounds, os.path.join(ENCODER_DIR, "clip_bounds.pkl"))
print(f"  Saved clip bounds → {ENCODER_DIR}clip_bounds.pkl")


print("\n[Step 8] Building Review Scores Composite (post-split)...")
review_sub_cols = [
    "Review Scores Accuracy",
    "Review Scores Cleanliness",
    "Review Scores Checkin",
    "Review Scores Communication",
    "Review Scores Location",
    "Review Scores Value",
]
available_review_cols = [c for c in review_sub_cols if c in X_train.columns]

if available_review_cols:
    # Compute row-wise mean on train (NaN rows stay NaN here)
    X_train["Review Scores Composite"] = X_train[available_review_cols].mean(axis=1)
    X_test["Review Scores Composite"] = X_test[available_review_cols].mean(axis=1)
    print(f"  Composite built from {len(available_review_cols)} sub-scores.")
else:
    print("  No review sub-score columns found.")


print("\n[Step 9a] Re-deriving demand_label_3 from train-only tertiles...")
if "demand_score" in df.columns:
    train_demand = df.loc[X_train.index, "demand_score"]
    test_demand = df.loc[X_test.index, "demand_score"]

    q33_train, q67_train = train_demand.quantile([1 / 3, 2 / 3]).values
    print(f"  Train tertiles: 33rd={q33_train:.4f}, 67th={q67_train:.4f}")

    def assign_3class(score, q33, q67):
        if score <= q33:
            return 0
        elif score <= q67:
            return 1
        else:
            return 2

    y_cls3_train = train_demand.apply(assign_3class, args=(q33_train, q67_train))
    y_cls3_test = test_demand.apply(assign_3class, args=(q33_train, q67_train))

    # Save thresholds for inference
    joblib.dump(
        {"q33": q33_train, "q67": q67_train},
        os.path.join(ENCODER_DIR, "demand_label_3_thresholds.pkl"),
    )
    print(f"  3-class — Train: {y_cls3_train.value_counts().sort_index().to_dict()}")
    print(f"  3-class — Test : {y_cls3_test.value_counts().sort_index().to_dict()}")
else:
    print("  'demand_score' not found — skipping re-derivation.")

# Re-derive demand_label (binary) from train-only median
print("\n[Step 9b] Re-deriving demand_label from train-only median...")
train_demand_scores = df.loc[X_train.index, "demand_score"]
test_demand_scores = df.loc[X_test.index, "demand_score"]

train_median_demand = train_demand_scores.median()
print(f"  Train-only demand_score median: {train_median_demand:.4f}")

y_cls_train = (train_demand_scores > train_median_demand).astype(int)
y_cls_test = (test_demand_scores > train_median_demand).astype(int)

# Save threshold for inference
joblib.dump(
    {"median": train_median_demand},
    os.path.join(ENCODER_DIR, "demand_label_threshold.pkl"),
)
print(f"  Binary — Train: {y_cls_train.value_counts().to_dict()}")
print(f"  Binary — Test : {y_cls_test.value_counts().to_dict()}")


print("\n[Step 10] Encoding categoricals...")

if "Parsed Amenities" in X_train.columns:
    print("  Extracting Top-20 Amenities from train...")
    amenity_counts = Counter()
    for row in X_train["Parsed Amenities"].dropna():
        amenity_counts.update(row.split("|"))

    top_20_amenities = [a for a, _ in amenity_counts.most_common(20) if a]
    joblib.dump(top_20_amenities, os.path.join(ENCODER_DIR, "top_20_amenities.pkl"))

    for amenity in top_20_amenities:
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", amenity)
        col_name = f"Amenity_{safe_name}"
        X_train[col_name] = (
            X_train["Parsed Amenities"]
            .str.contains(amenity, regex=False, na=False)
            .astype(int)
        )
        X_test[col_name] = (
            X_test["Parsed Amenities"]
            .str.contains(amenity, regex=False, na=False)
            .astype(int)
        )

    X_train.drop(columns=["Parsed Amenities"], inplace=True)
    X_test.drop(columns=["Parsed Amenities"], inplace=True)
    print(f"  Created {len(top_20_amenities)} amenity one-hot columns.")

categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

for col in categorical_cols:
    unique_count = X_train[col].nunique(dropna=False)

    if unique_count > 15:
        # LabelEncoder — fit on train, handle unseen in test as "Unknown"
        le = LabelEncoder()
        X_train[col] = X_train[col].fillna("Unknown").astype(str)
        X_test[col] = X_test[col].fillna("Unknown").astype(str)

        le.fit(X_train[col])
        known = set(le.classes_)

        if "Unknown" not in known:
            le.classes_ = np.append(le.classes_, "Unknown")

        X_test[col] = X_test[col].apply(lambda x: x if x in known else "Unknown")

        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

        joblib.dump(le, os.path.join(ENCODER_DIR, f"{col}_label_encoder.pkl"))

    else:
        # One-hot — fit structure on train, reindex test to match
        train_dummies = pd.get_dummies(X_train[col].fillna("Unknown"), prefix=col)
        test_dummies = pd.get_dummies(X_test[col].fillna("Unknown"), prefix=col)
        test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

        X_train = pd.concat(
            [X_train.drop(columns=[col]), train_dummies.astype(int)], axis=1
        )
        X_test = pd.concat(
            [X_test.drop(columns=[col]), test_dummies.astype(int)], axis=1
        )

# Save final feature column list
joblib.dump(X_train.columns.tolist(), os.path.join(ENCODER_DIR, "feature_columns.pkl"))
print(f"  Saved feature column list ({len(X_train.columns)} features).")


print("\n[Step 11] Median imputation (train-only, continuous cols only)...")

# Identify binary columns (one-hot / flag columns), exclude from median fill
binary_cols = [
    c
    for c in X_train.select_dtypes(include=[np.number]).columns
    if X_train[c].dropna().nunique() <= 2
]

continuous_cols = [
    c
    for c in X_train.select_dtypes(include=[np.number]).columns
    if c not in binary_cols
]

train_medians = X_train[continuous_cols].median()
X_train[continuous_cols] = X_train[continuous_cols].fillna(train_medians)
X_test[continuous_cols] = X_test[continuous_cols].fillna(train_medians)

# Binary columns: fill with train mode (most frequent value: 0 or 1)
train_modes = X_train[binary_cols].mode().iloc[0]
X_train[binary_cols] = X_train[binary_cols].fillna(train_modes)
X_test[binary_cols] = X_test[binary_cols].fillna(train_modes)

joblib.dump(train_medians, os.path.join(ENCODER_DIR, "train_medians.pkl"))
joblib.dump(train_modes, os.path.join(ENCODER_DIR, "train_modes.pkl"))
print(f"  Imputed {len(continuous_cols)} continuous cols with median.")
print(f"  Imputed {len(binary_cols)} binary cols with mode.")


print("\n[Step 12] VIF check...")
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
    high_vif = vif_df[vif_df["VIF"] > 10]["Feature"].tolist()
    if high_vif:
        print(
            f"  ⚠ High VIF (>10): {high_vif} — consider dropping before linear models."
        )


print("\n[Step 13] Scaling continuous features...")

# Re-identify binary cols after all encoding steps
binary_cols_final = [
    c
    for c in X_train.select_dtypes(include=[np.number]).columns
    if X_train[c].dropna().nunique() <= 2
]
scale_cols = [
    c
    for c in X_train.select_dtypes(include=[np.number]).columns
    if c not in binary_cols_final
]

scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

joblib.dump(scaler, os.path.join(ENCODER_DIR, "standard_scaler.pkl"))
joblib.dump(scale_cols, os.path.join(ENCODER_DIR, "scaled_columns.pkl"))
print(f"  Scaled {len(scale_cols)} continuous columns.")
print(f"  Left {len(binary_cols_final)} binary columns unscaled.")


print("\n[Step 14] Saving train/test splits...")

# Reset index so positional alignment is safe for .values assignment
train_orig_idx = X_train.index.copy()
test_orig_idx = X_test.index.copy()

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)


def build_output_df(X, y_reg, y_cls, y_cls3, orig_idx, raw_cols):
    out = X.copy()
    out[REGRESSION_TARGET] = y_reg.values
    out[CLASSIFICATION_TARGET_BIN] = y_cls.values
    out[CLASSIFICATION_TARGET_MULTI] = y_cls3.values
    out["Price_original"] = df.loc[orig_idx, "Price_original"].values
    for col, series in raw_cols.items():
        out[f"{col}_raw"] = series.reset_index(drop=True).values
    return out


train_df = build_output_df(
    X_train, y_reg_train, y_cls_train, y_cls3_train, train_orig_idx, raw_train
)
test_df = build_output_df(
    X_test, y_reg_test, y_cls_test, y_cls3_test, test_orig_idx, raw_test
)

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)


print("\n" + "=" * 60)
print("POSTPROCESSING COMPLETE — LEAKAGE-FREE")
print("=" * 60)
print(f"  Train rows          : {len(train_df):,}")
print(f"  Test rows           : {len(test_df):,}")
print(f"  Features            : {X_train.shape[1]}")
print(f"  Remaining nulls     : {X_train.isna().sum().sum()}")
print(f"  Encoders saved to   : {ENCODER_DIR}")
print(f"  Splits saved to     : {OUTPUT_DIR}")
print("=" * 60)
print("\nTargets in output CSVs:")
print(f"  Regression  → {REGRESSION_TARGET}  (use np.expm1() to recover dollars)")
print(f"  Binary cls  → {CLASSIFICATION_TARGET_BIN}   (0=low, 1=high demand)")
print(f"  3-class cls → {CLASSIFICATION_TARGET_MULTI}  (0=low, 1=med, 2=high)")
print("\nArtifacts saved:")
print("  clip_bounds.pkl               — outlier caps for all numeric cols + target")
print("  demand_label_3_thresholds.pkl — train-only tertile boundaries")
print("  top_20_amenities.pkl          — amenity list from train")
print("  feature_columns.pkl           — final column order")
print("  train_medians.pkl             — continuous imputation values")
print("  train_modes.pkl               — binary imputation values")
print("  standard_scaler.pkl           — fitted scaler")
print("  scaled_columns.pkl            — which columns were scaled")
print("  *_label_encoder.pkl           — one per high-cardinality categorical")
