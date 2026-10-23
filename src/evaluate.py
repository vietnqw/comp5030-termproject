"""Evaluate a checkpoint on a CSV split (used for the one-time NIH test).

    python -m src.evaluate --config configs/baseline.yaml --ckpt outputs/baseline/best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import LABELS, ChestXrayDataset
from src.model import build_model


def load_config(path, data_root=None, output_dir=None):
    with open(path) as f:
        config = yaml.safe_load(f)
    if data_root:
        config["data_root"] = data_root
    if output_dir:
        config["output_dir"] = output_dir
    return config


def make_loader(config, csv_key, shuffle=False):
    dataset = ChestXrayDataset(
        Path(config["data_root"]) / config[csv_key],
        limit=config.get("limit"),
    )
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers"],
        pin_memory=True,
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, targets = [], []
    for images, labels in tqdm(loader, desc="predict", leave=False):
        logits = model(images.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def auroc_table(probs, targets):
    """Per-class AUROC (None when a class has a single value) plus the mean."""
    table = {}
    for i, label in enumerate(LABELS):
        if len(np.unique(targets[:, i])) < 2:
            table[label] = None
        else:
            table[label] = roc_auc_score(targets[:, i], probs[:, i])
    valid = [v for v in table.values() if v is not None]
    table["mean"] = float(np.mean(valid)) if valid else None
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--csv-key", default="test_csv", help="config key naming the CSV to evaluate")
    parser.add_argument("--data-root")
    args = parser.parse_args()

    config = load_config(args.config, data_root=args.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))

    probs, targets = predict(model, make_loader(config, args.csv_key), device)
    table = auroc_table(probs, targets)

    for label, value in table.items():
        print(f"{label:15s} {value:.4f}" if value is not None else f"{label:15s} n/a")
    out_path = Path(args.ckpt).parent / f"{args.csv_key}_metrics.json"
    out_path.write_text(json.dumps(table, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
