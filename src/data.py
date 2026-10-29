from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Pathologies shared by CheXpert and NIH ChestX-ray14.
LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Pneumonia",
    "Pneumothorax",
]

# ImageNet statistics (DenseNet-121 is ImageNet-pretrained).
NORMALIZE = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
TRANSFORM = transforms.Compose([transforms.Resize((224, 224))] + NORMALIZE)
# Strong appearance augmentation for domain generalization. No horizontal flip:
# it breaks laterality (e.g. Cardiomegaly).
AUG_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
    ]
    + NORMALIZE
)


class ChestXrayDataset(Dataset):
    """CSV-driven dataset: `image_path` (relative to the CSV's directory) plus one 0/1 column per label."""

    def __init__(self, csv_path, limit=None, augment=False):
        self.df = pd.read_csv(csv_path)
        if limit:
            self.df = self.df.head(limit)
        self.root = Path(csv_path).parent
        self.transform = AUG_TRANSFORM if augment else TRANSFORM
        self.labels = self.df[LABELS].to_numpy(dtype="float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(self.root / self.df.iloc[idx]["image_path"]).convert("RGB")
        return self.transform(image), torch.from_numpy(self.labels[idx])
