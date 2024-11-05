import torch
import torch.nn as nn


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=5):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Ensure label range aligns with model's output classes
            if outputs.shape[1] <= torch.max(labels):
                labels = torch.clamp(labels, max=outputs.shape[1] - 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return model
