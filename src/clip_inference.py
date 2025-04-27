# CLIP-based inference pipeline for activity classification
import cv2
import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np
from collections import deque
import open_clip
import logging

# Configuration
MODEL_CONFIG = {
    'base_name': 'classifier_clip',  # Must match the name used in training
    'version': 'v1',                     # Must match the version used in training
    'full_model_name': None,             # Will be set automatically
}
MODEL_CONFIG['full_model_name'] = f"{MODEL_CONFIG['base_name']}_{MODEL_CONFIG['version']}.pth"

# Paths
model_path = f"./models/{MODEL_CONFIG['full_model_name']}"  # Path to trained model
data_dir = "./output/training_output"  # Same directory used for training

# Function to convert logits to probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Function to get smoothed prediction
def get_smoothed_prediction(prediction_buffer):
    if not prediction_buffer:
        return None
    # Average probabilities over the buffer
    avg_probs = np.mean(prediction_buffer, axis=0)
    return avg_probs

def run_clip_inference():
    # Initialize CLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device)

    # Load the model and prompts
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    classes = checkpoint['classes']
    class_prompts = checkpoint['class_prompts']
    model.eval()

    # Encode text prompts
    text_features = {}
    for class_name in classes:
        text = class_prompts[class_name]
        text_tokens = tokenizer([text]).to(device)
        with torch.no_grad():
            text_feature = model.encode_text(text_tokens)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            text_features[class_name] = text_feature

    # Temporal smoothing buffer
    buffer_size = 15  # Number of frames to consider for smoothing
    prediction_buffer = deque(maxlen=buffer_size)

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
            return

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

                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Get CLIP features
                with torch.no_grad():
                    image_features = model.encode_image(preprocess(pil_image).unsqueeze(0).to(device))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarities with all text prompts
                    similarities = {}
                    for class_name, text_feature in text_features.items():
                        similarity = (100.0 * image_features @ text_feature.T).softmax(dim=-1)
                        similarities[class_name] = similarity.item()
                    
                    # Convert to probabilities
                    probs = np.array(list(similarities.values()))
                    probs = softmax(probs)
                    prediction_buffer.append(probs)

                # Get smoothed prediction
                smoothed_probs = get_smoothed_prediction(prediction_buffer)
                if smoothed_probs is not None:
                    predicted_idx = np.argmax(smoothed_probs)
                    predicted_label = classes[predicted_idx]

                    # Display predictions
                    y_offset = 30
                    for i, (class_name, prob) in enumerate(zip(classes, smoothed_probs)):
                        prob_text = f"{class_name}: {prob*100:.1f}%"
                        color = (0, 255, 0) if i == predicted_idx else (255, 255, 255)
                        cv2.putText(frame, prob_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        y_offset += 30

                    cv2.imshow("CLIP-based Classification", frame)
                
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

if __name__ == "__main__":
    run_clip_inference() 