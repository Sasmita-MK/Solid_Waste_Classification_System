import torch
from ModelDefinition import create_model
from DatasetPreProcessing import load_data
from Training import train_model
from Webcam import capture_and_classify
import os


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=6).to(device)

    # Load or Train Model
    if os.path.exists("trained_model.pth"):
        model.load_state_dict(torch.load("trained_model.pth"))
        print("Loaded pretrained model.")
    else:
        _, dataloader = load_data(batch_size=32)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model = train_model(model, dataloader, criterion, optimizer, device, num_epochs=5)
        torch.save(model.state_dict(), "trained_model.pth")
        print("Training complete and model saved.")

    # Start Webcam Classification
    capture_and_classify(model, device)


if __name__ == "__main__":
    main()
