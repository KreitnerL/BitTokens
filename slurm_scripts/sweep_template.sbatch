#!/bin/bash

#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:a100:1 -C a100_80
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100

source /etc/profile
cd $2
source $2/.venv/bin/activate
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

python $2/train.py --load_config_from $3 --wandb_sweep_id $1 --save_dir $2/trained/sweep_$1
