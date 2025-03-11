#!/bin/bash
set -e

# Define paths
SCAN_PATH="data/scans"
OUTPUT_PATH="output"
MODEL_PATH="models/model.pth"

echo "Generating labels for scans in $SCAN_PATH..."
mkdir -p "$OUTPUT_PATH"

# Loop through all .bin LiDAR scan files
# for scan_file in "$SCAN_PATH"/*.bin; do
    # filename=$(basename -- "$scan_file")
    # output_file="$OUTPUT_PATH/${filename%.bin}.txt"

    # echo "Processing: $scan_file -> $output_file"

python3 kmeans_and_ml.py  --input $SCAN_PATH --output $OUTPUT_PATH --use_ml
done

echo "Labels saved in $OUTPUT_PATH"

