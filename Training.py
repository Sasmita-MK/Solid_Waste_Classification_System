import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=5):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Validate labels to prevent IndexError
            if torch.any(labels >= 6) or torch.any(labels < 0):
                print(f"Warning: Out-of-range label encountered in batch, skipping. Labels: {labels}")
                continue  # Skip this batch if invalid labels are found

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    print("Training completed.")
    return model