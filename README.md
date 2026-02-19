# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset

This experiment demonstrates transfer learning using a pre-trained ResNet18 model on a custom image dataset. Instead of training a deep neural network from scratch, the pre-trained model’s feature extraction layers are reused, and only the final classification layer is retrained. This approach reduces training time, requires less data, and achieves high accuracy.

## DESIGN STEPS
### STEP 1:
Data Preprocessing – Resize all images to 224×224 and convert them into tensors suitable for ResNet input.

### STEP 2: 
Dataset Loading – Organize images into train/test sets and load them using ImageFolder and DataLoader.

### STEP 3:
Load Pretrained Model – Use ResNet18 trained on ImageNet as the base model.

### STEP 4:
Modify Final Layer – Freeze earlier layers and replace the fully connected layer to match the number of dataset classes.

### STEP 5:
Train and Evaluate – Train only the final layer, then test the model and analyze results using a confusion matrix and classification report.

## PROGRAM
```python
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# Modify the final fully connected layer to match the dataset classes
for param in model.parameters():
    param.requires_grad = False   # freeze earlier layers
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

```
```python
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: Keerthivasan K S")
    print("Register Number:  212224230120")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="979" height="761" alt="image" src="https://github.com/user-attachments/assets/68dad096-ad21-4d02-9772-ae8138a49898" />

### Confusion Matrix
<img width="979" height="541" alt="image" src="https://github.com/user-attachments/assets/8696e857-d670-431e-92bc-fe0f8696662a" />

### Classification Report
<img width="488" height="176" alt="image" src="https://github.com/user-attachments/assets/891b1ebf-e707-4470-a8c5-b26bbeaa7db2" />

### New Sample Prediction
<img width="378" height="431" alt="image" src="https://github.com/user-attachments/assets/7ad41571-4952-488a-bcba-7d2539a74a20" />

## RESULT
The Implementation of Transfer Learning for classification using VGG-19 architecture is successful.
