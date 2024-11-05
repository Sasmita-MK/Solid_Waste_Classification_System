import os
import torch
from datasets import load_dataset, load_from_disk
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def load_data(batch_size=32, dataset_path="cached_dataset"):
    # Check if the dataset is already saved locally
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
        print("Loaded dataset from disk cache.")
    else:
        dataset = load_dataset("rootstrap-org/waste-classifier")
        dataset.save_to_disk(dataset_path)
        print("Dataset loaded and saved to disk.")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure image is in RGB format
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Custom PyTorch Dataset class
    class WasteDataset(Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            example = self.dataset[idx]
            image = example["image"]
            label = example["label"]

            # Apply transformations to the image
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)  # Ensure label is an integer tensor

    # Convert Hugging Face dataset split to PyTorch-compatible dataset
    train_dataset = WasteDataset(dataset["train"], transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, dataloader
