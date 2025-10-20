#!/bin/bash

declare -a G
GS=$(seq 1 10)

THREADS=15
SOLUTION_OUT_FOLDER="../out"
LOG_OUT_FOLDER="../log"
echo "instance, k, model, status, objval, runtime, nodes, optgap, dual"
# Loop through each combination of parameters
for G in $GS;
do
  for INSTANCE in "slashdot" "bitcoin" "epinions"
  do

    for STRUCT in "struct_clusters" "struct_intervals"
    do
      for CONFIG in "config_gaia" "config_venus"
      do
        INSTANCE_FILE=$(printf "../data/%s.json" "$INSTANCE")
        CONFIG_FILE=$(printf "../data/%s.json" "$CONFIG")
        STRUCT_FILE=$(printf "../data/%s.json" "$STRUCT")
        SOLUTION_FILE=$(printf "%s/%s_%s_%s_%s.json" "$SOLUTION_OUT_FOLDER" "$INSTANCE" "$CONFIG" "$STRUCT" "$G")
        LOG_FILE=$(printf "%s/%s_%s_%s_%s.csv" "$LOG_OUT_FOLDER" "$INSTANCE" "$CONFIG" "$STRUCT" "$G")

  #      python kmst.py --instance="$FILE" --model="$MOD" --k="${K1[$G]}" &
        {
          ../target/release/heuristics "$INSTANCE_FILE" "$STRUCT_FILE" "$CONFIG_FILE" "$SOLUTION_FILE" gaic --seed $G > $LOG_FILE 2>/dev/null; echo "$INSTANCE,$STRUCT,$CONFIG,$G,$(tail $LOG_FILE -n 1)"
        } &
        # Wait for the maximum number of parallel jobs to be reached
        while [ "$(jobs | wc -l)" -ge $THREADS ]; do
          sleep 1
        done
      done
    done
  done
done
wait
