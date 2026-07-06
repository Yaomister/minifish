#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out


cd $SLURM_SUBMIT_DIR
module load python/3.13.5

mkdir -p training
mkdir -p logs

source .venv/bin/activate

python3 training.py
