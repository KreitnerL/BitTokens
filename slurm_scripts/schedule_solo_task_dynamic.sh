#!/bin/bash

source .env

declare -A implementations
implementations["fe"]="fe"
implementations["sd"]="sd"
implementations["td"]="td"
implementations["FoNE"]="FoNE"
implementations["xVal"]="xVal"
# implementations["base10"]="base10"

declare -A tasks
tasks["Addition"]="Addition"
tasks["Multiplication"]="Multiplication"
tasks["Division"]="Division"
tasks["MinMax"]="MinMax"
tasks["Sorting"]="Sorting"
tasks["Interval"]="Interval"
tasks["Exponentiation"]="Exponentiation"
tasks["Mean"]="Mean"
tasks["Std"]="Std"
# tasks["Text"]="Text"

export PARAMS=""
suffix=$(echo "$PARAMS" | sed -E 's/--([a-zA-Z0-9_]+) ([^ ]+)/\1=\2-/g' | sed 's/- /\-/g' | sed 's/-$//')

for implementation in "${!implementations[@]}"; do
    for task in "${!tasks[@]}"; do
        TASK="${tasks[$task]}"
        echo "Configured TASK: $TASK"
        IMPLEMENTATION="${implementations[$implementation]}"
        echo "Submitting SoloTask job for implementation: $IMPLEMENTATION with params: $PARAMS"
        sbatch --job-name="SoloTask_${IMPLEMENTATION}_${TASK}_${suffix}" --output=$PROJECT_PATH/slurm/wandb/solotask-$IMPLEMENTATION-$TASK-$suffix-%A.out --error=$PROJECT_PATH/slurm/wandb/solotask-$IMPLEMENTATION-$TASK-$suffix-%A.err $PROJECT_PATH/slurm_scripts/solo_task_dynamic.sbatch "$IMPLEMENTATION" "$TASK" "$PARAMS"
        sleep 1
    done
done

echo "All jobs submitted!"
