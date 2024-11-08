# DatasetPreProcessing.py
import os
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from PIL import Image

def load_data(batch_size=32, dataset_path="cached_dataset"):
    # Force re-create the cached dataset by deleting the existing cache
    if os.path.exists(dataset_path):
        print("Deleting existing cached dataset...")
        import shutil
        shutil.rmtree(dataset_path)

    # Load and filter dataset
    dataset = load_dataset("rootstrap-org/waste-classifier")
    dataset = dataset.filter(lambda x: x["label"] in range(6))

    # Save the filtered dataset to ensure caching works correctly
    dataset.save_to_disk(dataset_path)
    print("Filtered dataset saved to disk.")

    # Enhanced Transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Custom Dataset class
    class WasteDataset(Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.dataset = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            example = self.dataset[idx]
            image = example["image"].convert("RGB")
            label = example["label"]

            # Check if label is out of expected range
            if label not in range(6):
                raise ValueError(f"Label out of range: {label}. Expected range: [0, 5]")

            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label)

    # Load dataset into WasteDataset and DataLoader
    train_dataset = WasteDataset(dataset["train"], transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataset, dataloader