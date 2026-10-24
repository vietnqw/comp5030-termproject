#!/bin/bash
#SBATCH --job-name=cxr-data
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# One-time: wipe any partial downloads, download + extract both datasets, build the CSVs.
set -euo pipefail
. ~/.cheaha_paths.sh
cd "$WORKSPACE/comp5030-termproject"

rm -rf "$DATASETS/chexpert-small" "$DATASETS/nih-cxr14"
bash scripts/download_data.sh

echo "== extracted layout =="
find "$DATASETS/chexpert-small" "$DATASETS/nih-cxr14" -maxdepth 3 -type d

uv run python scripts/prepare_data.py --data-root "$DATASETS"
echo "== done =="
