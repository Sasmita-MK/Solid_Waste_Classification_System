import torch
from ModelDefinition import create_resnet, create_mobilenet, create_densenet
from DatasetPreProcessing import load_data, calculate_class_weights
from Training import train_model
from Webcam import capture_and_classify
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet_model = create_resnet(num_classes=6).to(device)
    mobilenet_model = create_mobilenet(num_classes=6).to(device)
    densenet_model = create_densenet(num_classes=6).to(device)

    _, dataloader = load_data(batch_size=32)
    class_weights = calculate_class_weights(_)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    if os.path.exists("resnet_trained.pth"):
        resnet_model.load_state_dict(torch.load("resnet_trained.pth"))
        print("Loaded pretrained ResNet model.")
    else:
        optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.0001)
        train_model(resnet_model, dataloader, criterion, optimizer, device, num_epochs=10, model_path="resnet_trained.pth")

    if os.path.exists("mobilenet_trained.pth"):
        mobilenet_model.load_state_dict(torch.load("mobilenet_trained.pth"))
        print("Loaded pretrained MobileNet model.")
    else:
        optimizer = torch.optim.Adam(mobilenet_model.parameters(), lr=0.0001)
        train_model(mobilenet_model, dataloader, criterion, optimizer, device, num_epochs=10, model_path="mobilenet_trained.pth")

    if os.path.exists("densenet_trained.pth"):
        densenet_model.load_state_dict(torch.load("densenet_trained.pth"))
        print("Loaded pretrained DenseNet model.")
    else:
        optimizer = torch.optim.Adam(densenet_model.parameters(), lr=0.0001)
        train_model(densenet_model, dataloader, criterion, optimizer, device, num_epochs=10, model_path="densenet_trained.pth")

    models = {
        "ResNet": resnet_model,
        "MobileNet": mobilenet_model,
        "DenseNet": densenet_model
    }

    capture_and_classify(models, device)

if __name__ == "__main__":
    main()