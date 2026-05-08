#!/bin/bash

source .env

declare -A task_metrics
declare -A implementations
implementations["fe"]="fe"
implementations["sd"]="sd"
implementations["td"]="td"
implementations["FoNE"]="FoNE"
implementations["xVal"]="xVal"
implementations["base10"]="base10"

# export PARAMS="--freq_loss_scheduler cosine_repeat --loss_freq_decay_factor 0.8"
export PARAMS=""
suffix=$(echo "$PARAMS" | sed -E 's/--([a-zA-Z0-9_]+) ([^ ]+)/\1=\2-/g' | sed 's/- /\-/g' | sed 's/-$//')

for implementation in "${!implementations[@]}"; do
    IMPLEMENTATION="${implementations[$implementation]}"
    echo "Submitting SoloTask job for implementation: $IMPLEMENTATION with params: $PARAMS"
    sbatch --job-name="SoloTask_${IMPLEMENTATION}_${suffix}" --output=$PROJECT_PATH/slurm/wandb/solotask-$IMPLEMENTATION-$suffix-%A.out --error=$PROJECT_PATH/slurm/wandb/solotask-$IMPLEMENTATION-$suffix-%A.err $PROJECT_PATH/slurm_scripts/solo_task.sbatch "$IMPLEMENTATION" "$PARAMS"
    sleep 1
done

echo "All jobs submitted!"
