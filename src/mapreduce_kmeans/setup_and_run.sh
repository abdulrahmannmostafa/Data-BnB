#!/bin/bash
# ══════════════════════════════════════════════════════════════
# K-Means MapReduce — Complete Setup & Run Script
#
# Run this from the mapreduce_kmeans directory:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh
# ══════════════════════════════════════════════════════════════

set -e

K=4
MAX_ITER=20
TOL=0.0001
DATA_FILE="data_full.tsv"
HDFS_INPUT="/user/$USER/kmeans/input"

echo "═══════════════════════════════════════════"
echo " K-Means MapReduce — Setup & Run"
echo "═══════════════════════════════════════════"

# ── Step 1: Start Hadoop services ──
echo ""
echo "[Step 1] Starting Hadoop services..."
start-dfs.sh
start-yarn.sh
echo "Waiting 5 seconds for services to initialize..."
sleep 5

# Verify DataNode is running
if ! jps | grep -q "DataNode"; then
    echo "ERROR: DataNode not running!"
    echo "Try: rm -rf /usr/local/hadoop/hdfs/datanode/* && start-dfs.sh"
    exit 1
fi
echo "All services running:"
jps

# ── Step 2: Upload data to HDFS ──
echo ""
echo "[Step 2] Uploading data to HDFS..."
hdfs dfs -mkdir -p $HDFS_INPUT
hdfs dfs -put -f $DATA_FILE $HDFS_INPUT/data.tsv
echo "Uploaded $DATA_FILE to $HDFS_INPUT/data.tsv"
hdfs dfs -ls $HDFS_INPUT

# ── Step 3: Run K-Means MapReduce ──
echo ""
echo "[Step 3] Running K-Means MapReduce..."
./run_hadoop.sh $K $MAX_ITER $TOL

# ── Step 4: Done ──
echo ""
echo "═══════════════════════════════════════════"
echo " COMPLETE! Results in centroids.txt"
echo "═══════════════════════════════════════════"
cat centroids.txt
