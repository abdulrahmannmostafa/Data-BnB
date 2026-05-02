import pandas as pd

train_df = pd.read_csv("/home/amostafa/CMP/Data-BnB/data/splits/train.csv", nrows=2)
TARGET = "demand_label"
DROP_COLS = [
    "demand_label",
    "demand_label_3",
    "demand_score",
    "Price_log",
    "Price_original",
    "Price_vs_city_median",
]
raw_cols = [c for c in train_df.columns if c.endswith("_raw")]
amenity_cols = [c for c in train_df.columns if "Parsed Amenities" in c]
DROP_COLS += raw_cols + amenity_cols
feature_cols = [c for c in train_df.columns if c not in DROP_COLS]
print("Features:", feature_cols)
