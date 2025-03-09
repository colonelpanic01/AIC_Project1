#!/bin/bash
# Ensure the script stops on errors
set -e

# Define paths
SCAN_PATH="data/scans"
OUTPUT_PATH="output"
# MODEL_PATH="models/model.pth"  # Update with your TensorRT model file

echo "Generating labels for scans in $SCAN_PATH..."
mkdir -p "$OUTPUT_PATH"

# Loop through all .bin LiDAR scan files
for scan_file in "$SCAN_PATH"/*.bin; do
    filename=$(basename -- "$scan_file")
    output_file="$OUTPUT_PATH/${filename%.bin}.txt"
    
    echo "Processing: $scan_file -> $output_file"
    
    # Run inference (replace with your inference script or binary)
    python3 lidar_vru_detection.py  --use_ml
done

echo "Labels saved in $OUTPUT_PATH"

