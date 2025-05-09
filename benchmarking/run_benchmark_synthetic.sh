#!/bin/bash
#run from repo root
declare -a SEED
GS=$(seq 1 50)

THREADS=61
SOLUTION_OUT_FOLDER="benchmarking/synthetic_solutions"
LOG_OUT_FOLDER="benchmarking/synthetic_logs"
CONFIG_FOLDER="benchmarking/synthetic_configs/*"
STRUCT_FOLDER="benchmarking/synthetic_structs/*"
INSTANCE_FOLDER="/Data/Interval_Development/Synthetic_Datasets/*.json"
RUST_EXE="heuristics/target/release/heuristics"
echo "instance, config, struct, seed, edge_weight,current, best_batch, best,best_plus_negative_edge_weight, epochs_since_restart, current_temp, runtime_ms"
# Loop through each combination of parameters
for SEED in $GS;
do
  for INSTANCE in $INSTANCE_FOLDER
  do
    for STRUCT in $STRUCT_FOLDER
    do
      for CONFIG in $CONFIG_FOLDER
      do
#        INSTANCE_FILE=$(printf "%s/%s.json" "$INSTANCE")
#        CONFIG_FILE=$(printf "../data/%s.json" "$CONFIG")
#        STRUCT_FILE=$(printf "../data/%s.json" "$STRUCT")
        SOLUTION_FILE=$(printf "%s/%s_%s_%s_%s.json" "$SOLUTION_OUT_FOLDER" "${INSTANCE##*/}" "${CONFIG##*/}" "${STRUCT##*/}" "$SEED")
        LOG_FILE=$(printf "%s/%s_%s_%s_%s.csv" "$LOG_OUT_FOLDER" "${INSTANCE##*/}" "${CONFIG##*/}" "${STRUCT##*/}" "$SEED")

        {
          "$RUST_EXE" "$INSTANCE" "$STRUCT" "$CONFIG" "$SOLUTION_FILE" gaic --seed "$SEED" > "$LOG_FILE" 2>/dev/null ; echo "${INSTANCE##*/}", "${CONFIG##*/}", "${STRUCT##*/}", "$SEED", "$(tail $LOG_FILE -n 1)"
        } &
        # Wait for the maximum number of parallel jobs to be reached
        while [ "$(jobs | wc -l)" -ge $THREADS ]; do
          # echo "Waiting for jobs to finish..." # 2>/dev/null
          sleep 1
        done
      done
    done
  done
done
wait
