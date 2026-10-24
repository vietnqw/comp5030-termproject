# Chest X-ray Domain Generalization (COMP5030 Term Project)

Improve unseen-domain robustness for chest X-ray disease classification: train DenseNet-121
(ImageNet-pretrained) on CheXpert-small, test on NIH ChestX-ray14. NIH is never used for
training or model selection (selection = mean AUROC on a patient-disjoint CheXpert val split).

- **Labels** (shared by both datasets, multi-label): Atelectasis, Cardiomegaly, Consolidation,
  Edema, Effusion, Pneumonia, Pneumothorax. CheXpert uncertain (-1) labels count as negative.
- **Metric**: per-class AUROC + mean AUROC. NIH is evaluated once, on the official `test_list.txt` split.
- Current state: ERM baseline. DG techniques will be added as config switches.

## Layout

```
configs/    baseline.yaml (real runs), smoke.yaml (tiny CPU pipeline check)
src/        data.py, model.py, train.py, evaluate.py
scripts/    download_data.sh, prepare_data.py
jobs/       train.sbatch, eval.sbatch (Cheaha SLURM)
```

Outputs (checkpoints, metrics) are written wherever `--output-dir` points; nothing is committed.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/): `uv sync`.

Data goes under one root (`$DATASETS` on Cheaha, `./data/` locally):

```bash
bash scripts/download_data.sh                     # Kaggle mirrors; needs ~/.kaggle/kaggle.json
uv run python scripts/prepare_data.py --data-root "$DATASETS"
```

`prepare_data.py` writes the unified CSVs the code reads: `chexpert-small/chexpert_{train,val}.csv`
and `nih-cxr14/nih_test.csv` (image paths relative to each CSV's directory).

## Run

Local (any machine):

```bash
uv run python -m src.train --config configs/baseline.yaml --data-root data
uv run python -m src.evaluate --config configs/baseline.yaml --data-root data --ckpt outputs/baseline/best.pt
```

Cheaha (clone into `$WORKSPACE`, then from the repo root):

```bash
uv sync
mkdir -p logs
sbatch jobs/data.sbatch                  # one-time: download + extract + build CSVs
sbatch jobs/train.sbatch                 # -> $RUNS/comp5030-termproject/baseline-<jobid>
RUN=baseline-<jobid> sbatch jobs/eval.sbatch   # NIH test + copy results to $RESULTS
```
