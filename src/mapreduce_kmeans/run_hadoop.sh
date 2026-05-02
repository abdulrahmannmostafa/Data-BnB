#!/bin/bash
# ══════════════════════════════════════════════════════════════
# K-Means MapReduce — Hadoop Streaming Runner
#
# Prerequisites:
#   - Hadoop installed and HDFS running
#   - Data uploaded: hdfs dfs -put data.tsv /user/$USER/kmeans/input/
#   - mapper.py, reducer.py, centroids.txt in current directory
#
# Usage:
#   chmod +x run_hadoop.sh
#   ./run_hadoop.sh [K] [MAX_ITER] [TOL]
# ══════════════════════════════════════════════════════════════

K=${1:-4}
MAX_ITER=${2:-20}
TOL=${3:-0.0001}

HADOOP_STREAMING_JAR=$(find $HADOOP_HOME/share/hadoop/tools/lib -name "hadoop-streaming*.jar" | head -1)
INPUT_DIR="/user/$USER/kmeans/input"
OUTPUT_BASE="/user/$USER/kmeans/output"

echo "═══════════════════════════════════════════"
echo " K-Means MapReduce on Hadoop"
echo " K=$K, max_iter=$MAX_ITER, tol=$TOL"
echo "═══════════════════════════════════════════"

# Initialize centroids (pick K random lines from input)
echo "Initializing centroids..."
hdfs dfs -cat $INPUT_DIR/data.tsv | shuf -n $K | awk -F'\t' '{print NR-1 "\t" $0}' | tr '\t' ',' | \
  awk -F',' '{printf "%s\t", $1; for(i=2;i<=NF;i++) printf "%s%s", $i, (i<NF?",":""); print ""}' > centroids.txt

echo "Initial centroids:"
cat centroids.txt
echo ""

# Iterate
for ((iter=1; iter<=MAX_ITER; iter++)); do
    OUTPUT_DIR="${OUTPUT_BASE}/iter_${iter}"

    # Remove previous output if exists
    hdfs dfs -rm -r -f $OUTPUT_DIR 2>/dev/null

    echo "--- Iteration $iter ---"

    # Run Hadoop Streaming job
    hadoop jar $HADOOP_STREAMING_JAR \
        -files mapper.py,reducer.py,centroids.txt \
        -mapper "python3 mapper.py" \
        -reducer "python3 reducer.py" \
        -input $INPUT_DIR \
        -output $OUTPUT_DIR

    if [ $? -ne 0 ]; then
        echo "ERROR: Hadoop job failed at iteration $iter"
        exit 1
    fi

    # Get new centroids from output
    hdfs dfs -cat $OUTPUT_DIR/part-* > new_centroids.txt

    # Check convergence (max shift between old and new centroids)
    SHIFT=$(python3 -c "
import math
old = {}
new = {}
with open('centroids.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        old[int(parts[0])] = list(map(float, parts[1].split(',')))
with open('new_centroids.txt') as f:
    for line in f:
        parts = line.strip().split('\t')
        new[int(parts[0])] = list(map(float, parts[1].split(',')))
max_shift = 0
for k in old:
    if k in new:
        d = math.sqrt(sum((a-b)**2 for a,b in zip(old[k], new[k])))
        max_shift = max(max_shift, d)
print(f'{max_shift:.6f}')
")

    echo "  Max centroid shift: $SHIFT"

    # Update centroids
    cp new_centroids.txt centroids.txt

    # Check convergence
    CONVERGED=$(python3 -c "print(1 if $SHIFT < $TOL else 0)")
    if [ "$CONVERGED" == "1" ]; then
        echo ""
        echo "Converged after $iter iterations!"
        break
    fi
done

echo ""
echo "═══════════════════════════════════════════"
echo "Final centroids:"
cat centroids.txt
echo ""
echo "Saved to: centroids.txt"
echo "═══════════════════════════════════════════"
