import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
INPUT_PATH = "../data/airbnb-cleaned.csv"
OUTPUT_DIR = "../data/splits/"
ENCODER_DIR = "../encoders/"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)

# FIX: columns we want to keep in their original (unscaled) form so that
# descriptive_analysis.py can produce meaningful cluster profile stats.
# These are saved as  <col>_raw  alongside the scaled splits.
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

print("Loading cleaned dataset...")
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
# FIX 4 — Log-transform target
# ──────────────────────────────────────────────
print("\n[Fix 4] Log-transforming target...")
df = df[df["Price"].notna()]
df = df[df["Price"] > 0]

df["Price_log"] = np.log1p(df["Price"])

# ──────────────────────────────────────────────
# FIX 5 — Review score composite
# ──────────────────────────────────────────────
print("\n[Fix 5] Review score aggregation...")
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
    print(f"  Composite built from {len(available_review_cols)} columns.")

# ──────────────────────────────────────────────
# FIX 6 — Train/Test split BEFORE city-relative medians
#         Prevents leakage
# ──────────────────────────────────────────────
print("\n[Fix 6] Train/Test split...")
TARGET_COL = "Price_log"

DROP_FROM_FEATURES = ["Price", "Price_log"]

if "ID" in df.columns:
    DROP_FROM_FEATURES.append("ID")

X = df.drop(columns=[c for c in DROP_FROM_FEATURES if c in df.columns])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print(f"  Train: {X_train.shape}")
print(f"  Test : {X_test.shape}")

# ──────────────────────────────────────────────
# FIX 6b — Snapshot raw profile columns BEFORE any scaling/encoding
#           so descriptive_analysis.py can profile clusters on real values.
# ──────────────────────────────────────────────
print("\n[Fix 6b] Snapshotting raw profile columns before encoding/scaling...")

raw_train_cols = {
    col: X_train[col].copy() for col in RAW_PROFILE_COLS if col in X_train.columns
}
raw_test_cols = {
    col: X_test[col].copy() for col in RAW_PROFILE_COLS if col in X_test.columns
}

saved_raw = list(raw_train_cols.keys())
print(f"  Snapshotted {len(saved_raw)} columns: {saved_raw}")

# ──────────────────────────────────────────────
# FIX 7 — City-relative price from TRAIN ONLY
# ──────────────────────────────────────────────
print("\n[Fix 7] City-relative price feature (train-only medians)...")

city_col = None
for candidate in ["City", "Neighbourhood Cleansed"]:
    if candidate in X_train.columns:
        city_col = candidate
        break

if city_col:
    train_city_medians = (
        X_train.groupby(city_col)["Price"].median()
        if "Price" in X_train.columns
        else df.loc[X_train.index].groupby(city_col)["Price"].median()
    )

    X_train["Price_vs_city_median"] = df.loc[X_train.index, "Price"] / X_train[
        city_col
    ].map(train_city_medians)

    global_train_price_median = df.loc[X_train.index, "Price"].median()

    X_test["Price_vs_city_median"] = df.loc[X_test.index, "Price"] / X_test[
        city_col
    ].map(train_city_medians).fillna(global_train_price_median)

    X_train["Price_vs_city_median"] = X_train["Price_vs_city_median"].fillna(1.0)
    X_test["Price_vs_city_median"] = X_test["Price_vs_city_median"].fillna(1.0)

    joblib.dump(train_city_medians, os.path.join(ENCODER_DIR, "train_city_medians.pkl"))

    print(f"  Created using '{city_col}'.")

else:
    print("  No city column found — skipped.")

# ──────────────────────────────────────────────
# FIX 8 — Encode categoricals AFTER split
# ──────────────────────────────────────────────
print("\n[Fix 8] Encoding categoricals...")

categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

saved_dummy_columns = []

for col in categorical_cols:
    unique_count = X_train[col].nunique(dropna=False)

    if unique_count > 15:
        le = LabelEncoder()

        X_train[col] = X_train[col].fillna("Unknown").astype(str)
        X_test[col] = X_test[col].fillna("Unknown").astype(str)

        le.fit(X_train[col])

        known_classes = set(le.classes_)
        X_test[col] = X_test[col].apply(
            lambda x: x if x in known_classes else "Unknown"
        )

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

        saved_dummy_columns.extend(train_dummies.columns.tolist())

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
# FIX 11 — Scaling for linear/regularized models
# ──────────────────────────────────────────────
print("\n[Fix 11] Feature scaling...")

scaler = StandardScaler()

numeric_cols = X_train.select_dtypes(include=[np.number]).columns

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

joblib.dump(scaler, os.path.join(ENCODER_DIR, "standard_scaler.pkl"))

# ──────────────────────────────────────────────
# FIX 12 — Save splits with raw profile columns re-attached
# ──────────────────────────────────────────────
print("\n[Fix 12] Saving splits...")

train_df = X_train.copy()
train_df[TARGET_COL] = y_train
train_df["Price_original"] = df.loc[X_train.index, "Price"]

# FIX: re-attach the pre-scaling snapshots as  <col>_raw  columns so that
# descriptive_analysis.py always has interpretable values for profiling,
# regardless of whether StandardScaler has transformed the main columns.
for col, series in raw_train_cols.items():
    train_df[f"{col}_raw"] = series.values

test_df = X_test.copy()
test_df[TARGET_COL] = y_test
test_df["Price_original"] = df.loc[X_test.index, "Price"]

for col, series in raw_test_cols.items():
    test_df[f"{col}_raw"] = series.values

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

print("\nSaved for inference:")
print("- train_medians.pkl")
print("- train_city_medians.pkl")
print("- feature_columns.pkl")
print("- standard_scaler.pkl")
print("- label encoders")
print(f"- {len(saved_raw)} raw profile columns (suffix _raw) in train/test CSVs")
print("\nUse np.expm1(predictions) to restore dollar prices.")
