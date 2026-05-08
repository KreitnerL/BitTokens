#!/bin/bash

source .env
declare -A implementations
implementations["sd"]="sd"
implementations["td"]="td"
implementations["FoNE"]="FoNE"
implementations["xVal"]="xVal"

export PARAMS=""
suffix=$(echo "$PARAMS" | sed -E 's/--([a-zA-Z0-9_]+) ([^ ]+)/\1=\2-/g' | sed 's/- /\-/g' | sed 's/-$//')

for implementation in "${!implementations[@]}"; do
    IMPLEMENTATION="${implementations[$implementation]}"
    echo "Submitting Multitask job for implementation: $IMPLEMENTATION with params: $PARAMS"
    sbatch --job-name="MultiTask_binary_${IMPLEMENTATION}_${suffix}" --output=$PROJECT_PATH/slurm/wandb/multitask-$IMPLEMENTATION-$suffix-%A.out --error=$PROJECT_PATH/slurm/wandb/multitask-$IMPLEMENTATION-$suffix-%A.err $PROJECT_PATH/slurm_scripts/multi_task_binary.sbatch "$IMPLEMENTATION" "$PARAMS"
    sleep 1
done

echo "All jobs submitted!"

