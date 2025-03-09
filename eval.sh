#!/bin/bash

TEAM_NAME='team_1'

# your execution path
EXEC_PATH='test_exec.sh'


SCAN_PATH="$PWD/test_of_test/scans/"
LABEL_PATH="$PWD/test_of_test/labels/"

mkdir "$PWD/output_$TEAM_NAME/"

OUTPUT_PATH="$PWD/output_$TEAM_NAME/"

# This is where you run the model

start=`date +%s.%N`
source $EXEC_PATH --scan_path $SCAN_PATH  --output_path $OUTPUT_PATH
end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )

echo "Runtime: $runtime"

python3 map_eval_3d.py --label_path $LABEL_PATH --output_path $OUTPUT_PATH
