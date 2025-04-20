# Run image classification on DroidCam feed using custom trained model

import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

# Paths
model_path = "./models/5activity_classifier.pth"  # Path to your trained model
data_dir = "./output/training_output"  # Same directory used for training

# Load the model architecture
model = models.mobilenet_v2(pretrained=False)  # Start with untrained model
# Get the number of classes from the dataset directory
num_classes = len(os.listdir(data_dir))
# Modify the final layer to match your number of classes
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)

# Load the trained weights
model.load_state_dict(torch.load(model_path))
model.eval()

# Get class names from the dataset directory
classes = sorted(os.listdir(data_dir))

# Image preprocessing (should match training preprocessing)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Try to open DroidCam (usually device 1 or 2)
for device_id in [1, 2]:
    cap = cv2.VideoCapture(device_id)
    if cap.isOpened():
        # Set some properties to ensure proper color format
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"Successfully connected to DroidCam on device {device_id}")
        break
else:
    print("Error: Could not connect to DroidCam. Make sure it's running and connected.")
    exit()

print("DroidCam started. Press 'q' to quit.")

# Function to convert logits to probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from DroidCam.")
        break

    # Check if frame is valid
    if frame is None or frame.size == 0:
        print("Error: Invalid frame received.")
        continue

    # Convert color space if needed
    if len(frame.shape) == 2:  # If grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:  # If RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Preprocess the frame
    try:
        input_tensor = transform(frame).unsqueeze(0)
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        continue

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        # Convert logits to probabilities
        probabilities = softmax(output.numpy()[0])
        # Get top prediction
        predicted_idx = np.argmax(probabilities)
        predicted_label = classes[predicted_idx]

    # Display predictions
    y_offset = 30
    for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
        # Format the probability as percentage
        prob_text = f"{class_name}: {prob*100:.1f}%"
        # Use different colors for the top prediction
        color = (0, 255, 0) if i == predicted_idx else (255, 255, 255)
        cv2.putText(frame, prob_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30

    cv2.imshow("DroidCam Classification", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()