import os
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from PIL import Image

def load_data(batch_size=32, dataset_path="cached_dataset"):
    # Check for cached dataset
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
        print("Loaded dataset from disk cache.")
    else:
        dataset = load_dataset("rootstrap-org/waste-classifier")

        # Filter dataset for valid labels (0 to 5)
        dataset = dataset.filter(lambda x: x["label"] in range(6))

        # Save the filtered dataset
        dataset.save_to_disk(dataset_path)
        print("Filtered dataset loaded and saved to disk.")

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
            image = example["image"].convert("RGB")  # Convert to RGB
            label = example["label"]
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label)

    train_dataset = WasteDataset(dataset["train"], transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataset, dataloader