#!/bin/bash

#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:a100:1 -C a100_80
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100

source .env
source /etc/profile
cd $PROJECT_PATH
source $PROJECT_PATH/.venv/bin/activate
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

suffix=$(echo "$3" | sed -E 's/--([a-zA-Z0-9_]+) ([^ ]+)/\1=\2-/g' | sed 's/- /\-/g' | sed 's/-$//')
export TASK=$2

python $PROJECT_PATH/train.py --load_config_from $PROJECT_PATH/configs/config_${1}_soloTask_dynamic.py $3 --run_name "$TASK-$suffix"
