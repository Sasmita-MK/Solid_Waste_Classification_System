import os
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def load_data(batch_size=32, dataset_path="cached_dataset"):
    if os.path.exists(dataset_path):
        print("Loading cached dataset...")
        dataset = load_from_disk(dataset_path)
    else:
        print("Downloading and caching dataset...")
        dataset = load_dataset("rootstrap-org/waste-classifier")
        dataset = dataset.filter(lambda x: x["label"] in range(6))
        dataset.save_to_disk(dataset_path)
        print("Filtered dataset saved to disk.")

    transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

            if label not in range(6):
                raise ValueError(f"Label out of range: {label}. Expected range: [0, 5]")

            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label)

    train_dataset = WasteDataset(dataset, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataset, dataloader

def calculate_class_weights(dataset):
    labels = []

    for sample in dataset:
        if isinstance(sample, tuple):
            _, label = sample
        elif isinstance(sample, dict):
            label = sample['label']
        else:
            raise ValueError("Unexpected data format. Each sample should be a tuple or dictionary.")

        labels.append(label)

    label_counts = np.bincount(labels, minlength=6)
    class_weights = 1.0 / (label_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()
    return torch.tensor(class_weights, dtype=torch.float32)