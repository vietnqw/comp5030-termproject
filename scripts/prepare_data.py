"""Build unified CSVs (image_path + one 0/1 column per shared label) for both datasets.

    python scripts/prepare_data.py --data-root $DATASETS

Expects (nested layout may vary; files are located by name):
  <data-root>/chexpert-small/.../{train.csv,valid.csv} + the images they reference
  <data-root>/nih-cxr14/.../{Data_Entry_2017.csv,test_list.txt} + *.png images
Writes:
  <data-root>/chexpert-small/{chexpert_train.csv,chexpert_val.csv}   (frontal only, U->0, patient-disjoint split)
  <data-root>/nih-cxr14/nih_test.csv                                 (official test_list.txt)
"""

import argparse
import sys
import zlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import LABELS  # noqa: E402

# CheXpert column name -> shared label name (others are identical).
CHEXPERT_RENAME = {"Pleural Effusion": "Effusion"}
VAL_FRACTION_DENOM = 20  # 1 in 20 patients -> val (~5%)


def locate(root, name):
    matches = list(root.rglob(name))
    if not matches:
        raise SystemExit(f"{name} not found anywhere under {root}")
    return matches[0]


def chexpert_path_fixer(root, base, sample):
    """CSV Path values look like 'CheXpert-v1.0-small/train/...'; find how they map onto
    the actual extracted layout and return a function rewriting them relative to root."""
    for strip in (False, True):
        rel = sample.split("/", 1)[1] if strip else sample
        if (base / rel).is_file():
            return lambda p: str((base / (p.split("/", 1)[1] if strip else p)).relative_to(root))
    raise SystemExit(f"cannot resolve image path {sample!r} against {base}")


def prepare_chexpert(root):
    base = locate(root, "train.csv").parent
    frames = {}
    for split in ("train", "valid"):
        df = pd.read_csv(base / f"{split}.csv")
        df = df[df["Frontal/Lateral"] == "Frontal"].rename(columns=CHEXPERT_RENAME)
        out = pd.DataFrame({"image_path": df["Path"]})
        for label in LABELS:
            # Uncertain (-1) and blank labels count as negative (U-zeros policy).
            out[label] = (df[label] == 1).astype(int)
        frames[split] = out

    fixer = chexpert_path_fixer(root, base, frames["train"]["image_path"].iloc[0])
    for out in frames.values():
        out["image_path"] = out["image_path"].map(fixer)

    patient = frames["train"]["image_path"].str.extract(r"(patient\d+)", expand=False)
    is_val = patient.map(lambda p: zlib.crc32(p.encode()) % VAL_FRACTION_DENOM == 0)
    train = frames["train"][~is_val]
    # Official valid.csv is tiny (~200 studies); fold it into our val split.
    val = pd.concat([frames["train"][is_val], frames["valid"]], ignore_index=True)

    train.to_csv(root / "chexpert_train.csv", index=False)
    val.to_csv(root / "chexpert_val.csv", index=False)
    print(f"chexpert: {len(train)} train, {len(val)} val")


def prepare_nih(root):
    image_paths = {p.name: str(p.relative_to(root)) for p in root.rglob("*.png")}
    test_names = set(locate(root, "test_list.txt").read_text().split())
    df = pd.read_csv(locate(root, "Data_Entry_2017.csv"))
    df = df[df["Image Index"].isin(test_names)]

    out = pd.DataFrame({"image_path": df["Image Index"].map(image_paths)})
    findings = df["Finding Labels"].str.split("|")
    for label in LABELS:
        out[label] = findings.map(lambda f: int(label in f)).to_numpy()
    missing = out["image_path"].isna().sum()
    if missing:
        raise SystemExit(f"nih: {missing} test images listed in test_list.txt not found on disk")

    out.to_csv(root / "nih_test.csv", index=False)
    print(f"nih: {len(out)} test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    prepare_chexpert(data_root / "chexpert-small")
    prepare_nih(data_root / "nih-cxr14")


if __name__ == "__main__":
    main()
