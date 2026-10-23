import torch.nn as nn
from torchvision import models

from src.data import LABELS


def build_model(pretrained=True):
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    return model
