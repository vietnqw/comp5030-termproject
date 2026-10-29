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
. ~/.cheaha_paths.sh
cd "$WORKSPACE/comp5030-termproject"
: "${RUN:?set RUN=<run-name> (folder under \$RUNS/comp5030-termproject)}"
RUN_DIR="$RUNS/comp5030-termproject/$RUN"

uv run python -m src.evaluate --config configs/baseline.yaml \
    --data-root "$DATASETS" \
    --ckpt "$RUN_DIR/best.pt"

# Keep the finished run's metrics + checkpoint on /data (scratch is purged).
mkdir -p "$PROJECTS/comp5030-termproject/$RUN"
cp "$RUN_DIR"/*.json "$RUN_DIR/best.pt" "$PROJECTS/comp5030-termproject/$RUN/"
