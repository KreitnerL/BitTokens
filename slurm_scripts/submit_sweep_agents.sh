#!/bin/bash

# Import environment variables
source .env

# Check if sweep ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <sweep_id> <number_of_agents> <config_file>"
    exit 1
fi

SWEEP_ID=$1
NUM_AGENTS=${2}
CONFIG_FILE=${3}

echo "Submitting $NUM_AGENTS Slurm jobs for W&B sweep $SWEEP_ID"

# Create the logs directory if it doesn't exist
mkdir -p $PROJECT_PATH/slurm/wandb

# Submit the jobs
for i in $(seq 1 $NUM_AGENTS); do
    echo "Submitting agent $i"
    sbatch -J sweep_$SWEEP_ID\_$i --output=$PROJECT_PATH/slurm/wandb/$SWEEP_ID\_$i-%A.out --error=$PROJECT_PATH/slurm/wandb/$SWEEP_ID\_$i-%A.err $PROJECT_PATH/slurm_scripts/sweep_template.sbatch $SWEEP_ID $PROJECT_PATH $CONFIG_FILE
done

echo "All jobs submitted!"
