# Multi-stage inference pipeline with YOLOv8 tracking, pose detection, and temporal smoothing
# Feature Vector Structure (14 dimensions):
# [0-5]    First object features (x1_norm, y1_norm, x2_norm, y2_norm, conf, cls)
# [6-11]   Second object features (x1_norm, y1_norm, x2_norm, y2_norm, conf, cls)
# [12-16]  Object type counts (up to 5 different object types)
# Pose features are currently disabled

import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Configuration
MODEL_CONFIG = {
    'base_name': 'activity_classifier_nopose',  # Must match the name used in training
    'version': 'v2',                     # Must match the version used in training
    'full_model_name': None,             # Will be set automatically
    'interrupted_model_name': None       # Will be set automatically
}

# Set model names
MODEL_CONFIG['full_model_name'] = f"{MODEL_CONFIG['base_name']}_{MODEL_CONFIG['version']}.pth"
MODEL_CONFIG['interrupted_model_name'] = f"{MODEL_CONFIG['base_name']}_{MODEL_CONFIG['version']}_interrupted.pth"

# Paths
model_path = f"./models/{MODEL_CONFIG['full_model_name']}"  # Path to trained model
data_dir = "./output/training_output"  # Same directory used for training

# Initialize YOLOv8 models
tracker = YOLO('yolov8m.pt', verbose=False)  # Using medium model for better detection
pose_detector = YOLO('yolov8n-pose.pt', verbose=False)  # Keep nano for pose as it's sufficient

# Constants
FEATURE_DIM = 17  # 12 for objects + 5 for counts
NUM_CLASSES = 5  # Adjust based on number of activity classes

# Define the YOLOv8 feature classifier architecture (must match training)
class YOLOv8Classifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(YOLOv8Classifier, self).__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# Load the activity classifier
num_classes = len(os.listdir(data_dir))
model = YOLOv8Classifier(FEATURE_DIM, num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Get class names
classes = sorted(os.listdir(data_dir))

# Temporal smoothing buffer
buffer_size = 15  # Number of frames to consider for smoothing
prediction_buffer = deque(maxlen=buffer_size)

# Function to convert logits to probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Function to get smoothed prediction
def get_smoothed_prediction():
    if not prediction_buffer:
        return None
    # Average probabilities over the buffer
    avg_probs = np.mean(prediction_buffer, axis=0)
    return avg_probs

# Function to extract features from YOLOv8 outputs
def extract_features(track_results, pose_results, frame):
    # Initialize feature vector
    features = np.zeros(FEATURE_DIM)
    
    # Get frame dimensions for normalization
    height, width = frame.shape[:2]
    
    # Track object counts and types
    object_counts = {}
    object_features = []
    
    if track_results[0].boxes is not None:
        # Get all detected objects
        for box in track_results[0].boxes:
            # Extract bounding box coordinates and confidence
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = tracker.names[cls]
            
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

try:
    # Try to open DroidCam
    for device_id in [1, 2]:
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"Successfully connected to DroidCam on device {device_id}")
            break
    else:
        print("Error: Could not connect to DroidCam. Make sure it's running and connected.")
        exit()

    print("DroidCam started. Press 'q' to quit.")

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from DroidCam.")
                break

            # Check if frame is valid
            if frame is None or frame.size == 0:
                continue

            # Convert color space if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Run YOLOv8 tracking and pose detection in parallel
            track_results = tracker.track(frame, persist=True)
            pose_results = pose_detector(frame)

            # Print detected objects for debugging
            if track_results[0].boxes is not None:
                print("\nDetected objects:")
                for box in track_results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = tracker.names[cls]
                    print(f"- {label} (confidence: {conf:.2f})")

            # Extract features from YOLOv8 outputs
            features = extract_features(track_results, pose_results, frame)
            print("\nExtracted features:", features)

            # Get activity prediction
            with torch.no_grad():
                # Convert features to tensor and add batch dimension
                input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                output = model(input_tensor)
                # Print raw model outputs before softmax
                print("\nRaw model outputs:", output.numpy()[0])
                probabilities = softmax(output.numpy()[0])
                prediction_buffer.append(probabilities)

            # Get smoothed prediction
            smoothed_probs = get_smoothed_prediction()
            if smoothed_probs is not None:
                predicted_idx = np.argmax(smoothed_probs)
                predicted_label = classes[predicted_idx]

                # Display predictions with more detail
                y_offset = 30
                print("\nClass probabilities:")
                for i, (class_name, prob) in enumerate(zip(classes, smoothed_probs)):
                    prob_text = f"{class_name}: {prob*100:.1f}%"
                    print(f"- {prob_text}")
                    color = (0, 255, 0) if i == predicted_idx else (255, 255, 255)
                    cv2.putText(frame, prob_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    y_offset += 30

                # Draw tracking results with bounding boxes and IDs
                annotated_frame = track_results[0].plot(
                    conf=True,  # Show confidence scores
                    line_width=2,  # Thicker lines for better visibility
                    font_size=1,  # Larger font size
                    boxes=True,  # Show bounding boxes
                    labels=True,  # Show class labels
                    probs=True  # Show probabilities
                )

                # Draw pose detection results on the same frame
                annotated_frame = pose_results[0].plot(
                    img=annotated_frame,  # Use the already annotated frame
                    conf=True,  # Show confidence scores
                    line_width=2,  # Thicker lines for better visibility
                    font_size=1,  # Larger font size
                    boxes=True,  # Show bounding boxes
                    labels=True,  # Show class labels
                    probs=True,  # Show probabilities
                    kpt_line=True,  # Show keypoint connections
                    kpt_radius=3  # Size of keypoint circles
                )

                # Add a title to the window
                cv2.putText(annotated_frame, "Multi-stage Classification", (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Multi-stage Classification", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            print("\nInference interrupted by user. Exiting gracefully...")
            break
        except Exception as e:
            print(f"\nAn error occurred during inference: {e}")
            continue

except KeyboardInterrupt:
    print("\nProgram interrupted by user. Exiting gracefully...")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    # Cleanup
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Resources cleaned up.")