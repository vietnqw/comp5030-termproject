#!/bin/bash
#SBATCH --job-name=cxr-data
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# One-time: download + extract both datasets, build the CSVs.
set -euo pipefail
. ~/.cheaha_env.sh

bash scripts/download_data.sh

echo "== extracted layout =="
find data/chexpert-small/ data/nih-cxr14/ -maxdepth 3 -type d

uv run python scripts/prepare_data.py --data-root data
echo "== done =="
