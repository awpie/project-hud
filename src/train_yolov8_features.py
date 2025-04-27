# Train a classifier using YOLOv8 features extracted from training images
# Feature Vector Structure (14 dimensions):
# [0-5]    First object features (x1_norm, y1_norm, x2_norm, y2_norm, conf, cls)
# [6-11]   Second object features (x1_norm, y1_norm, x2_norm, y2_norm, conf, cls)
# [12-16]  Object type counts (up to 5 different object types)
# Pose features are currently disabled

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Configuration
MODEL_CONFIG = {
    'base_name': 'activity_classifier_nopose',  # Base name for the model
    'version': 'v2',                     # Version identifier
    'full_model_name': None,             # Will be set automatically
    'interrupted_model_name': None       # Will be set automatically
}

# Initialize YOLOv8 model
yolo_model = YOLO('yolov8m.pt', verbose=False)  # Using medium model for better detection

# Constants
FEATURE_DIM = 17  # 12 for objects + 5 for counts
NUM_CLASSES = 5  # Adjust based on number of activity classes
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# Set model names
MODEL_CONFIG['full_model_name'] = f"{MODEL_CONFIG['base_name']}_{MODEL_CONFIG['version']}.pth"
MODEL_CONFIG['interrupted_model_name'] = f"{MODEL_CONFIG['base_name']}_{MODEL_CONFIG['version']}_interrupted.pth"

class YOLOv8FeatureDataset(Dataset):
    def __init__(self, data_dir, yolo_model, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.yolo_model = yolo_model
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
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Get YOLOv8 features
        results = self.yolo_model(image)
        
        # Extract features and create fixed-size feature vector
        feature_vector = self._extract_features(results)
        
        return torch.tensor(feature_vector, dtype=torch.float32), label

    def _extract_features(self, results):
        # Initialize feature vector
        features = np.zeros(FEATURE_DIM)
        
        # Get frame dimensions for normalization
        height, width = results[0].orig_img.shape[:2]
        
        # Track object counts and types
        object_counts = {}
        object_features = []
        
        if results[0].boxes is not None:
            # Get all detected objects
            for box in results[0].boxes:
                # Extract bounding box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.yolo_model.names[cls]
                
                # Normalize coordinates
                x1_norm = x1 / width
                y1_norm = y1 / height
                x2_norm = x2 / width
                y2_norm = y2 / height
                
                # Create a feature representation for this object
                obj_features = np.array([
                    x1_norm, y1_norm, x2_norm, y2_norm,  # Normalized bbox coordinates
                    conf,                                # Confidence score
                    cls                                  # Class ID
                ])
                
                object_features.append(obj_features)
                
                # Count object types
                if label in object_counts:
                    object_counts[label] += 1
                else:
                    object_counts[label] = 1
        
        # Sort objects by confidence and take top 2
        if object_features:
            object_features.sort(key=lambda x: x[4], reverse=True)  # Sort by confidence
            top_objects = object_features[:2]  # Take top 2 objects
            
            # Fill feature vector with top object features
            for i, obj in enumerate(top_objects):
                start_idx = i * 6
                features[start_idx:start_idx + 6] = obj
        
        # Add object counts as features
        count_features = np.zeros(5)  # Assuming max 5 different object types
        for i, (label, count) in enumerate(object_counts.items()):
            if i < 5:
                count_features[i] = count
        features[12:17] = count_features
        
        return features

class YOLOv8Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(YOLOv8Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

def train_model():
    # Paths
    data_dir = "./output/training_output"
    model_output_dir = "./models"
    os.makedirs(model_output_dir, exist_ok=True)

    try:
        # Create dataset
        dataset = YOLOv8FeatureDataset(data_dir, yolo_model)
        
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = YOLOv8Classifier(FEATURE_DIM, NUM_CLASSES).to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(NUM_EPOCHS):
            try:
                # Training
                model.train()
                train_loss = 0.0
                for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                    features, labels = features.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * features.size(0)
                
                train_loss = train_loss / len(train_loader.dataset)
                
                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for features, labels in val_loader:
                        features, labels = features.to(device), labels.to(device)
                        outputs = model(features)
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
                    torch.save(model.state_dict(), os.path.join(model_output_dir, MODEL_CONFIG['full_model_name']))
                    print(f"  New best model saved as {MODEL_CONFIG['full_model_name']}!")
            
            except KeyboardInterrupt:
                print("\nTraining interrupted by user. Saving current model state...")
                # Save the current model state
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, os.path.join(model_output_dir, MODEL_CONFIG['interrupted_model_name']))
                print(f"Model state saved as {MODEL_CONFIG['interrupted_model_name']}. You can resume training later.")
                break

    except KeyboardInterrupt:
        print("\nTraining setup interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'train_loader' in locals():
            del train_loader
        if 'val_loader' in locals():
            del val_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("Training resources cleaned up.")

if __name__ == "__main__":
    train_model() 