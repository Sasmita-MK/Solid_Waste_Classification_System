import torch.nn as nn
import torchvision.models as models

def create_mobilenet(num_classes=6):
    model = models.mobilenet_v3_large(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def create_densenet(num_classes=6):
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def create_resnet(num_classes=6):
    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model