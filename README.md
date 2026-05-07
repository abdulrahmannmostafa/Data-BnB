# Data-BnB

Data-BnB is a big data analytics project focused on Airbnb listings. It combines:

- **Data preparation and feature engineering** for listings data
- **Demand labeling** for classification tasks
- **Leakage-safe postprocessing** and train/test artifact generation
- **Predictive modeling experiments** in notebooks and Python modules
- **MapReduce K-Means** implementation (local simulation + Hadoop Streaming)

---

## Repository Structure

```text
Data-BnB/
├── docs/                      # Project reports, presentation, and supporting material
├── encoders/                  # Saved preprocessing artifacts (*.pkl)
├── src/
│   ├── notebooks/             # EDA and modeling notebooks
│   ├── scripts/               # Core preprocessing, labeling, and ML modules
│   └── mapreduce_kmeans/      # K-Means with mapper/reducer + Hadoop scripts
└── README.md
```

---

## Main Workflow (Data Pipeline)

The core tabular pipeline in `src/scripts/` is designed to run in the following order:

1. **Preprocess raw data**
   - Script: `src/scripts/preprocessing.py`
   - Input: `../../data/airbnb-listings.csv`
   - Output: `../../data/airbnb-cleaned.csv`

2. **Engineer demand labels**
   - Script: `src/scripts/label_engineering.py`
   - Input: `../../data/airbnb-cleaned.csv`
   - Output: `../../data/airbnb-labeled.csv`

3. **Postprocess and split data (leakage-safe)**
   - Script: `src/scripts/postprocessing.py`
   - Input: `../../data/airbnb-labeled.csv`
   - Outputs:
     - `../../data/splits/train.csv`
     - `../../data/splits/test.csv`
     - Preprocessing artifacts under `../../encoders/`

---

## MapReduce K-Means Module

The `src/mapreduce_kmeans/` directory includes a full K-Means implementation from scratch using the MapReduce paradigm:

- `mapper.py` — assigns each point to the nearest centroid
- `reducer.py` — computes new centroids (cluster means)
- `driver.py` — local iterative simulation (`mapper | sort | reducer`)
- `run_hadoop.sh` — Hadoop Streaming iterative runner
- `prepare_data.py` — exports and optionally standardizes clustering input features
- `validate_comparison.py` — compares MapReduce centroids/labels with scikit-learn K-Means

For details and usage examples, see:
- `src/mapreduce_kmeans/README.md`

---

## Notebooks

Exploration and experimentation are under `src/notebooks/`, including:

- EDA and descriptive analysis
- Classification model notebooks (logistic regression, SVM, KNN, tree-based, boosting, Naive Bayes, MLP)
- Price-focused experimentation notebooks

---

## Environment and Dependencies

This repository does not currently include a pinned dependency file (`requirements.txt` or `pyproject.toml`), so install dependencies manually.

### Core Python packages used across scripts

- `pandas`
- `numpy`
- `scikit-learn`
- `statsmodels`
- `joblib`
- `matplotlib`
- `seaborn`
- `optuna`
- `xgboost`
- `catboost`
- `lightgbm`

Example installation:

```bash
pip install pandas numpy scikit-learn statsmodels joblib matplotlib seaborn optuna xgboost catboost lightgbm
```

---

## Data Notes

- The scripts expect Airbnb CSV files under a top-level `data/` directory.
- The repository `.gitignore` excludes `data/**/**`, so raw/processed datasets are not tracked by Git.

---

## How to Run (Quick Start)

From the repository root:

```bash
cd src/scripts
python preprocessing.py
python label_engineering.py
python postprocessing.py
```

For MapReduce K-Means local simulation:

```bash
cd src/mapreduce_kmeans
python prepare_data.py --sample 30000
python driver.py --k 4 --max-iter 20 --tol 0.0001
```

---

## Project Goal

The objective of Data-BnB is to support data-driven Airbnb strategy by combining:

- pricing and demand analytics,
- predictive modeling,
- and scalable clustering workflows aligned with big data processing concepts.
