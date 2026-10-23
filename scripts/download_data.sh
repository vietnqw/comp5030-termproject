#!/bin/bash
# Download both datasets into $DATASETS (run once, on a login node or your laptop).
# Needs a Kaggle API token at ~/.kaggle/kaggle.json (kaggle.com -> Settings -> Create New Token).
# NOTE: these are community Kaggle mirrors, not the official Stanford/NIH distribution pages;
# verify them before relying on the data (official sources: stanfordmlgroup.github.io/competitions/chexpert,
# nihcc.app.box.com/v/ChestXray-NIHCC).
set -euo pipefail
: "${DATASETS:?source ~/.cheaha_paths.sh first (or export DATASETS=...)}"

# CheXpert-small (~11 GB): downscaled CheXpert-v1.0
mkdir -p "$DATASETS/chexpert-small"
uvx kaggle datasets download -d ashery/chexpert -p "$DATASETS/chexpert-small" --unzip

# NIH ChestX-ray14 (~45 GB)
mkdir -p "$DATASETS/nih-cxr14"
uvx kaggle datasets download -d nih-chest-xrays/data -p "$DATASETS/nih-cxr14" --unzip
