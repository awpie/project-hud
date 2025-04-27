# Train a classifier using OpenCLIP features for direct image-text comparison
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import open_clip
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import logging

# Configuration
MODEL_CONFIG = {
    'base_name': 'classifier_clip',  # Base name for the model
    'version': 'v1',                     # Version identifier
    'full_model_name': None,             # Will be set automatically
    'interrupted_model_name': None       # Will be set automatically
}

# Constants
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# Set model names
MODEL_CONFIG['full_model_name'] = f"{MODEL_CONFIG['base_name']}_{MODEL_CONFIG['version']}.pth"
MODEL_CONFIG['interrupted_model_name'] = f"{MODEL_CONFIG['base_name']}_{MODEL_CONFIG['version']}_interrupted.pth"

class CLIPFeatureDataset(Dataset):
    def __init__(self, data_dir, clip_model, clip_preprocess):
        self.data_dir = data_dir
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect all image paths and their labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[class_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        
        # Get CLIP features
        image_features = self.clip_model.encode_image(self.clip_preprocess(image).unsqueeze(0))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.squeeze(), label

def train_model():
    # Paths
    data_dir = "./output/training_output"
    model_output_dir = "./models"
    os.makedirs(model_output_dir, exist_ok=True)

    try:
        # Initialize CLIP model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        model = model.to(device)
        
        # Create dataset
        dataset = CLIPFeatureDataset(data_dir, model, preprocess)
        
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize classifier
        feature_dim = 512  # CLIP ViT-B-32 feature dimension
        num_classes = len(dataset.classes)
        classifier = nn.Linear(feature_dim, num_classes).to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(NUM_EPOCHS):
            try:
                # Training
                classifier.train()
                train_loss = 0.0
                for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                    features, labels = features.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = classifier(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * features.size(0)
                
                train_loss = train_loss / len(train_loader.dataset)
                
                # Validation
                classifier.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for features, labels in val_loader:
                        features, labels = features.to(device), labels.to(device)
                        outputs = classifier(features)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * features.size(0)
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                val_loss = val_loss / len(val_loader.dataset)
                accuracy = 100 * correct / total
                
                print(f"Epoch {epoch+1}:")
                print(f"  Training Loss: {train_loss:.4f}")
                print(f"  Validation Loss: {val_loss:.4f}")
                print(f"  Validation Accuracy: {accuracy:.2f}%")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'classifier_state_dict': classifier.state_dict(),
                        'classes': dataset.classes
                    }, os.path.join(model_output_dir, MODEL_CONFIG['full_model_name']))
                    print(f"  New best model saved as {MODEL_CONFIG['full_model_name']}!")
            
            except KeyboardInterrupt:
                print("\nTraining interrupted by user. Saving current model state...")
                # Save the current model state
                torch.save({
                    'epoch': epoch,
                    'classifier_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'classes': dataset.classes
                }, os.path.join(model_output_dir, MODEL_CONFIG['interrupted_model_name']))
                print(f"Model state saved as {MODEL_CONFIG['interrupted_model_name']}. You can resume training later.")
                break

    except KeyboardInterrupt:
        print("\nTraining setup interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Cleanup
        if 'classifier' in locals():
            del classifier
        if 'train_loader' in locals():
            del train_loader
        if 'val_loader' in locals():
            del val_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("Training resources cleaned up.")

if __name__ == "__main__":
    train_model() 