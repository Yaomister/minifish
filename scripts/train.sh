#!/bin/bash
#SBATCH --job-name=parser
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/parse_%j.out


cd $SLURM_SUBMIT_DIR
mkdir -p training
mkdir -p logs
source .venv/bin/activate
python3 -m training.py
