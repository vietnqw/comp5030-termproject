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

. ~/.cheaha_paths.sh
cd "$WORKSPACE/comp5030-termproject"
RUN=${RUN:-baseline-$SLURM_JOB_ID}

uv run python -m src.train --config configs/baseline.yaml \
    --data-root "$DATASETS" \
    --output-dir "$RUNS/comp5030-termproject/$RUN"
