#!/bin/bash
# Assumes you're in /Data/Interval_Development/ScalableCorrelationClustering/benchmarking

GS=$(seq 1 50)
THREADS=1
# SOLUTION_BASE_FOLDER="/Data/Interval_Development/ScalableCorrelationClustering/benchmarking/solutions"
LOG_OUT_FOLDER="/Data/Interval_Development/RAMA/benchmarking/logs"
# CONFIG_FOLDER="benchmarking/configs/*"
# STRUCT_FOLDER="benchmarking/structs/*"
INSTANCE_FOLDER="/Data/Interval_Development/Datasets/*.txt"
CPP_EXE="/Data/Interval_Development/RAMA/build/src/rama_text_input"
# echo "instance, seed, energy, runtime"
# Loop through each combination of parameters
for SEED in $GS;
do
  for INSTANCE in $INSTANCE_FOLDER; 
  do
    LOG_FILE=$(printf "%s/rama_%s_%s.csv" "$LOG_OUT_FOLDER" "${INSTANCE##*/}" "$SEED")

    {
      "$CPP_EXE" -f "$INSTANCE" > "$LOG_FILE" 2>/dev/null
    } 
    # # Wait for the maximum number of parallel jobs to be reached
    # while [ "$(jobs | wc -l)" -ge $THREADS ]; do
    #   # echo "Waiting for jobs to finish..."
    #   sleep 1
    # done
  done
done
wait
