"""Train DenseNet-121 on CheXpert; model selection by mean AUROC on the CheXpert val split.

    python -m src.train --config configs/baseline.yaml [--data-root ...] [--output-dir ...]
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.evaluate import auroc_table, load_config, make_loader, predict
from src.model import build_model


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total, count = 0.0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(images)
        count += len(images)
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_config(args.config, data_root=args.data_root, output_dir=args.output_dir)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = make_loader(config, "train_csv", train=True)
    val_loader = make_loader(config, "val_csv")
    model = build_model(pretrained=config["pretrained"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    history, best_auroc = [], 0.0
    for epoch in range(1, config["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_table = auroc_table(*predict(model, val_loader, device))
        auroc = val_table["mean"] or 0.0
        history.append({"epoch": epoch, "train_loss": loss, "val": val_table})
        print(f"epoch {epoch}: loss {loss:.4f}, val mean AUROC {auroc:.4f}")
        if auroc >= best_auroc:
            best_auroc = auroc
            torch.save(model.state_dict(), output_dir / "best.pt")
        (output_dir / "metrics.json").write_text(json.dumps(history, indent=2))

    print(f"best val mean AUROC {best_auroc:.4f}, checkpoint at {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
