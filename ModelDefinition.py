import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes=6):
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model