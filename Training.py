import torch

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=5, model_path="trained_model.pth"):
    model.train()
    best_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
            print(f"Best model saved with loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping activated.")
                break

    print("Training completed.")
    return model