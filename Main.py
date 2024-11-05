import torch
import torch.nn as nn
import torch.optim as optim
from Training import train_model
from ModelDefinition import initialize_model
from DatasetPreProcessing import load_data
from Webcam import capture_and_classify

def main():
    # Load and prepare data
    print("Loading dataset...")
    _, dataloader = load_data(batch_size=32)
    print("Dataset loaded.")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define model, loss, and optimizer
    model = initialize_model(num_classes=4)  # Assuming 4 waste categories
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("Starting model training...")
    model = train_model(model, dataloader, criterion, optimizer, device, num_epochs=5)
    print("Model training completed.")

    # Start webcam for classification
    print("Starting webcam for real-time classification...")
    capture_and_classify(model, device)

if __name__ == "__main__":
    main()