import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# 1. Load raw data
# ──────────────────────────────────────────────
INPUT_PATH = r"../data/archive/airbnb-listings.csv"
OUTPUT_PATH = r"../data/airbnb-cleaned.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT_PATH, sep=";", low_memory=False)
print(f"Raw shape: {df.shape}")

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
    # Too sparse (>79 % null)
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
print(f"After dropping {len(cols_to_drop)} columns: {df.shape}")

# ──────────────────────────────────────────────
# 3. Parse the Amenities column
#    - Extract count of amenities
#    - One-hot encode top-20 amenities
# ──────────────────────────────────────────────
print("Parsing amenities...")


def parse_amenities(val):
    if pd.isna(val):
        return []
    # Strip curly braces and quotes if present
    val = val.strip("{}")
    return [a.strip().strip('"') for a in val.split(",") if a.strip()]


amenity_lists = df["Amenities"].apply(parse_amenities)
df["Amenities Count"] = amenity_lists.apply(len)

# Find top 20 most common amenities
from collections import Counter

all_amenities = Counter()
for lst in amenity_lists:
    all_amenities.update(lst)
top_20 = [a for a, _ in all_amenities.most_common(20)]

for amenity in top_20:
    col_name = "Amenity_" + amenity.replace(" ", "_").replace("/", "_")
    df[col_name] = amenity_lists.apply(lambda lst, a=amenity: int(a in lst))

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


df["Host Verifications Count"] = df["Host Verifications"].apply(count_verifications)
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
reference_date = pd.Timestamp("2017-04-02")  # approximate scrape date

if "Host Since" in df.columns:
    df["Host Since"] = pd.to_datetime(df["Host Since"], errors="coerce")
    df["Host Tenure Days"] = (reference_date - df["Host Since"]).dt.days
    df.drop(columns=["Host Since"], inplace=True)

if "First Review" in df.columns:
    df["First Review"] = pd.to_datetime(df["First Review"], errors="coerce")
    df["Days Since First Review"] = (reference_date - df["First Review"]).dt.days
    df.drop(columns=["First Review"], inplace=True)

if "Last Review" in df.columns:
    df["Last Review"] = pd.to_datetime(df["Last Review"], errors="coerce")
    df["Days Since Last Review"] = (reference_date - df["Last Review"]).dt.days
    df.drop(columns=["Last Review"], inplace=True)

# ──────────────────────────────────────────────
# 7. Encode categorical columns
# ──────────────────────────────────────────────
print("Encoding categoricals...")

# Label-encode high-cardinality categoricals
from sklearn.preprocessing import LabelEncoder

label_encode_cols = [
    "Host Response Time",
    "Neighbourhood",
    "Neighbourhood Cleansed",
    "City",
    "State",
    "Zipcode",
    "Market",
    "Country",
    "Jurisdiction Names",
    "Host Location",
    "Host Neighbourhood",
]

for col in label_encode_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# One-hot encode low-cardinality categoricals
onehot_cols = [
    "Property Type",
    "Room Type",
    "Bed Type",
    "Country Code",
    "Cancellation Policy",
]

for col in onehot_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

df = pd.get_dummies(df, columns=onehot_cols, drop_first=True, dtype=int)

# ──────────────────────────────────────────────
# 8. Handle missing values
# ──────────────────────────────────────────────
print("Handling missing values...")

# Numerical columns: fill with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# ──────────────────────────────────────────────
# 9. Outlier capping (price, min/max nights)
# ──────────────────────────────────────────────
print("Capping outliers...")


def cap_outliers(series, lower_pct=0.01, upper_pct=0.99):
    lower = series.quantile(lower_pct)
    upper = series.quantile(upper_pct)
    return series.clip(lower, upper)


for col in ["Price", "Minimum Nights", "Maximum Nights",
            "Extra People", "Cleaning Fee", "Security Deposit"]:
    if col in df.columns:
        df[col] = cap_outliers(df[col])

# Remove rows where Price is 0
before = len(df)
df = df[df["Price"] > 0]
print(f"Removed {before - len(df)} rows with Price = 0")

# ──────────────────────────────────────────────
# 10. Save cleaned dataset
# ──────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nFinal shape: {df.shape}")
print(f"Saved to: {OUTPUT_PATH}")

# Quick summary
print("\n=== Column types summary ===")
print(df.dtypes.value_counts())
print(f"\nRemaining nulls: {df.isna().sum().sum()}")
