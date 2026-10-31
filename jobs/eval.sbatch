#!/bin/bash
#SBATCH --job-name=cxr-eval
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# One-time NIH test of a finished run: RUN=<run-name> sbatch jobs/eval.sbatch
. ~/.cheaha_env.sh
: "${RUN:?set RUN=<run-name> (folder under scratch/)}"
RUN_DIR="scratch/$RUN"

uv run python -m src.evaluate --config configs/baseline.yaml --ckpt "$RUN_DIR/best.pt"

# Keep the finished run's metrics + checkpoint in data/ (scratch is purged).
mkdir -p "data/$RUN"
cp "$RUN_DIR"/*.json "$RUN_DIR/best.pt" "data/$RUN/"
