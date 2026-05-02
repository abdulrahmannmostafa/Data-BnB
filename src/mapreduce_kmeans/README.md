# K-Means MapReduce Implementation

Custom K-Means clustering implemented from scratch using the MapReduce paradigm.

## Files

| File | Purpose |
|------|---------|
| `mapper.py` | Assigns each data point to nearest centroid |
| `reducer.py` | Computes new centroid (mean of assigned points) |
| `driver.py` | Local simulation: iterates mapper→sort→reducer until convergence |
| `prepare_data.py` | Exports clustering features from cleaned CSV to TSV for HDFS |
| `run_hadoop.sh` | Shell script to run on actual Hadoop cluster |
| `data.tsv` | 30k sample (standardized) — for local testing |
| `data_full.tsv` | Full 495k dataset (standardized) — for Hadoop |
| `centroids.txt` | Current/final centroids (updated each iteration) |
| `scaler_params.tsv` | StandardScaler mean/std for inverse transform |

## Algorithm

```
1. Initialize K centroids (random data points)
2. REPEAT:
     MAP:    Each point → (nearest_cluster_id, point_features)
     REDUCE: Each cluster → new_centroid = mean(all assigned points)
     Check convergence (max centroid shift < tolerance)
3. UNTIL converged or max_iterations reached
```

## Local Testing

```bash
# Prepare data (30k sample, standardized)
python prepare_data.py --sample 30000

# Run locally (simulates MapReduce with pipes)
python driver.py --k 4 --max-iter 20 --tol 0.0001
```

## Running on Hadoop

```bash
# 1. Upload data to HDFS
hdfs dfs -mkdir -p /user/$USER/kmeans/input
hdfs dfs -put data_full.tsv /user/$USER/kmeans/input/data.tsv

# 2. Run the iterative streaming job
chmod +x run_hadoop.sh
./run_hadoop.sh 4 20 0.0001

# Or manually (single iteration):
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
  -files mapper.py,reducer.py,centroids.txt \
  -mapper "python3 mapper.py" \
  -reducer "python3 reducer.py" \
  -input /user/$USER/kmeans/input \
  -output /user/$USER/kmeans/output/iter_1
```

## Features Used

Same 7 features as the sklearn implementation (standardized):
1. Latitude
2. Longitude
3. Price
4. Accommodates
5. Bedrooms
6. Bathrooms
7. Review Scores Rating

## Results (Local Test, K=4, 30k sample)

Converged in 18 iterations. Centroids are in standardized space —
use `scaler_params.tsv` to convert back to original scale.
