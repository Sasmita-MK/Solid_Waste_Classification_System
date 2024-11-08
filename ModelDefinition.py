import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes=6):
    model = models.resnet50(pretrained=True)  # Use ResNet50 as an example
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Set output layer for 6 classes
    return model