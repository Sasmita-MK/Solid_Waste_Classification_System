import torch
import torchvision.models as models
from torch import nn

def initialize_model(num_classes):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model