import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# 1. Load raw data
# ──────────────────────────────────────────────
INPUT_PATH = r"../../data/airbnb-listings.csv"
OUTPUT_PATH = r"../../data/airbnb-cleaned.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT_PATH, sep=";", low_memory=False)
print(f"Raw shape: {df.shape}")

# ──────────────────────────────────────────────
# 1.5 Parse currency columns EARLY
#    Required before any numeric operations
# ──────────────────────────────────────────────
print("Parsing currency columns...")

currency_cols = ["Price", "Cleaning Fee", "Security Deposit", "Extra People"]

for col in currency_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .replace("nan", np.nan)
            .replace("", np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ──────────────────────────────────────────────
# 2. Drop columns with no analytical value
# ──────────────────────────────────────────────
cols_to_drop = [
    # URLs
    "Listing Url",
    "Host URL",
    "Thumbnail Url",
    "Medium Url",
    "Picture Url",
    "XL Picture Url",
    "Host Thumbnail Url",
    "Host Picture Url",

    # Free-text
    "Summary",
    "Space",
    "Description",
    "Neighborhood Overview",
    "Notes",
    "Transit",
    "Access",
    "Interaction",
    "House Rules",
    "Host About",

    # Scrape / metadata
    "Scrape ID",
    "Last Scraped",
    "Calendar last Scraped",
    "Calendar Updated",

    # Too sparse
    "Square Feet",
    "License",
    "Has Availability",
    "Host Acceptance Rate",
    "Neighbourhood Group Cleansed",

    # Redundant
    "Geolocation",
    "Smart Location",
    "Street",
    "Host Total Listings Count",

    # Near-zero variance
    "Experiences Offered",

    # Identifiers / PII
    "Name",
    "Host Name",
]

df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
print(f"After dropping columns: {df.shape}")

# ──────────────────────────────────────────────
# 3. Parse Amenities
#    NOTE:
#    We keep Amenities Count only here.
#    Top-N one-hot should happen AFTER train/test split
#    to avoid leakage.
# ──────────────────────────────────────────────
print("Parsing amenities...")


def parse_amenities(val):
    if pd.isna(val):
        return []
    val = val.strip("{}")
    return [a.strip().strip('"') for a in val.split(",") if a.strip()]


if "Amenities" in df.columns:
    amenity_lists = df["Amenities"].apply(parse_amenities)

    # Basic robust feature
    df["Amenities Count"] = amenity_lists.apply(len)

    # Keep raw parsed string for postprocessing if needed
    df["Parsed Amenities"] = amenity_lists.apply(lambda x: "|".join(x))

    df.drop(columns=["Amenities"], inplace=True)

# ──────────────────────────────────────────────
# 4. Parse Host Verifications → count
# ──────────────────────────────────────────────
print("Parsing host verifications...")


def count_verifications(val):
    if pd.isna(val):
        return 0
    val = val.strip("[]'\"")
    return len([v.strip() for v in val.split(",") if v.strip()])


if "Host Verifications" in df.columns:
    df["Host Verifications Count"] = df["Host Verifications"].apply(
        count_verifications
    )
    df.drop(columns=["Host Verifications"], inplace=True)

# ──────────────────────────────────────────────
# 5. Parse Features column → boolean flags
# ──────────────────────────────────────────────
print("Parsing features...")

if "Features" in df.columns:
    df["Host Has Profile Pic"] = (
        df["Features"].str.contains("Host Has Profile Pic", na=False).astype(int)
    )

    df["Host Identity Verified"] = (
        df["Features"].str.contains("Host Identity Verified", na=False).astype(int)
    )

    df["Is Location Exact"] = (
        df["Features"].str.contains("Is Location Exact", na=False).astype(int)
    )

    df.drop(columns=["Features"], inplace=True)

# ──────────────────────────────────────────────
# 6. Parse date columns → derived features
# ──────────────────────────────────────────────
print("Parsing dates...")

reference_date = pd.Timestamp("2017-04-02")

date_cols = {
    "Host Since": "Host Tenure Days",
    "First Review": "Days Since First Review",
    "Last Review": "Days Since Last Review",
}

for original_col, new_col in date_cols.items():
    if original_col in df.columns:
        df[original_col] = pd.to_datetime(df[original_col], errors="coerce")
        df[new_col] = (reference_date - df[original_col]).dt.days
        df.drop(columns=[original_col], inplace=True)

# ──────────────────────────────────────────────
# 7. Handle categorical columns
#    IMPORTANT:
#    No LabelEncoder here to avoid inconsistent mappings
#    We only fill missing values.
#    Encoding should happen AFTER split in postprocessing.
# ──────────────────────────────────────────────
print("Preparing categorical columns...")

categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# ──────────────────────────────────────────────
# 8. NO median imputation here
#    Prevent train-test leakage
# ──────────────────────────────────────────────
print("Skipping median imputation (handled post-split)...")

# ──────────────────────────────────────────────
# 9. NO outlier capping here
#    Prevent train-test leakage
# ──────────────────────────────────────────────
print("Skipping outlier capping (handled post-split)...")

# ──────────────────────────────────────────────
# 10. Remove invalid target rows
#    Price must exist and be > 0
# ──────────────────────────────────────────────
if "Price" in df.columns:
    before = len(df)

    df = df[df["Price"].notna()]
    df = df[df["Price"] > 0]

    removed = before - len(df)
    print(f"Removed {removed} rows with invalid Price")

# ──────────────────────────────────────────────
# 11. Save cleaned dataset
# ──────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nFinal shape: {df.shape}")
print(f"Saved to: {OUTPUT_PATH}")

# ──────────────────────────────────────────────
# 12. Summary
# ──────────────────────────────────────────────
print("\n=== Column Types Summary ===")
print(df.dtypes.value_counts())

print(f"\nRemaining nulls: {df.isna().sum().sum()}")

# Optional null report
null_summary = df.isna().sum()
null_summary = null_summary[null_summary > 0].sort_values(ascending=False)

if len(null_summary) > 0:
    print("\n=== Columns Still Containing Nulls ===")
    print(null_summary)
else:
    print("\nNo remaining nulls.")