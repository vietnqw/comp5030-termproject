#!/bin/bash
#SBATCH --job-name=cxr-train
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# Pick the experiment with CONFIG, e.g.: CONFIG=configs/aug.yaml sbatch jobs/train.sbatch
. ~/.cheaha_env.sh
CONFIG=${CONFIG:-configs/baseline.yaml}
RUN=${RUN:-$(basename "$CONFIG" .yaml)-$SLURM_JOB_ID}

uv run python -m src.train --config "$CONFIG" --output-dir "scratch/$RUN"
