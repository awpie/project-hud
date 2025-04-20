# Train a custom image classifier using your own dataset (organized by folders)

import os
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import signal
import sys

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    global shutdown_flag
    print("\nReceived shutdown signal. Finishing current epoch and saving model...")
    shutdown_flag = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

# Paths
data_dir = "./output/training_output"  # structure: class1/, class2/, ...
model_output_dir = "./models"  # Directory to save the model
model_name = "5activity_classifier.pth"  # Name of the model file

# Create model output directory if it doesn't exist
os.makedirs(model_output_dir, exist_ok=True)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset and split into train and validation
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)  # 80% for training
val_size = dataset_size - train_size  # 20% for validation

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# Model
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))
model = model.to(device)  # Move model to GPU if available

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
patience = 3  # Number of epochs to wait before early stopping
patience_counter = 0

try:
    for epoch in range(20):  # Increased epochs since we have early stopping
        if shutdown_flag:
            print("\nGraceful shutdown initiated. Saving current model...")
            model_path = os.path.join(model_output_dir, f"interrupted_{model_name}")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to: {model_path}")
            break

        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            if shutdown_flag:
                break
                
            # Move data to GPU if available
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        if shutdown_flag:
            continue
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                if shutdown_flag:
                    break
                    
                # Move data to GPU if available
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        if shutdown_flag:
            continue
            
        val_loss = val_loss / len(val_loader.dataset)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}:")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = os.path.join(model_output_dir, model_name)
            torch.save(model.state_dict(), model_path)
            print(f"  New best model saved to: {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Saving current model state...")
    model_path = os.path.join(model_output_dir, f"error_{model_name}")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

finally:
    print("\nTraining complete or interrupted!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if device.type == "cuda":
        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"Final GPU memory reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    
    # Clean up GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()