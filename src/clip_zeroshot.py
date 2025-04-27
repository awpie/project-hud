# Zero-shot CLIP classification using direct image-to-text comparison
import cv2
import torch
from PIL import Image
import numpy as np
from collections import deque
import open_clip
import logging

# Define text prompts for each class

CLASS_PROMPTS = {
    'coding': [
        'point-of-view of coding on a computer',
        'person typing on a keyboard',
        'computer screen with code',
        'programming on a laptop'
    ],
    'eating': [
        'point-of-view of eating food',
        'person eating a meal',
        'dining at a table',
        'consuming food'
    ],
    'reading': [
        'point-of-view of reading a book',
        'person reading a book',
        'holding and reading a book',
        'looking at a book'
    ],
    'man': [
        'male human',
        'man',
        'male person',
        'adult male'
    ],
    'woman': [
        'female human',
        'woman',
        'female person',
        'adult female'
    ],
    'manypeople': [
        'group of people',
        'crowd of people',
        'multiple people',
        'several people together'
    ],
    'piano': [
        'playing the piano',
        'person at a piano',
        'piano keyboard',
        'musician playing piano'
    ],
    'nature': [
        'nature',
        'outdoor scene',
        'natural landscape',
        'outdoors'
    ],
    'bed': [
        'bed',
        'bedroom bed',
        'sleeping bed',
        'mattress'
    ],
    'table': [
        'table',
        'dining table',
        'desk table',
        'wooden table'
    ],
    'chair': [
        'chair',
        'sitting chair',
        'office chair',
        'dining chair'
    ],
    'desk': [
        'desk',
        'office desk',
        'writing desk',
        'computer desk'
    ],
    'bookshelf': [
        'bookshelf',
        'bookcase',
        'shelf with books',
        'library shelf'
    ],
    'kitchen': [
        'kitchen',
        'cooking area',
        'kitchen room',
        'food preparation area'
    ],
    'bottle': [
        'bottle',
        'water bottle',
        'drink bottle',
        'plastic bottle'
    ],
    'cup': [
        'cup',
        'drinking cup',
        'coffee cup',
        'tea cup'
    ],
    'bowl': [
        'bowl',
        'food bowl',
        'soup bowl',
        'cereal bowl'
    ],
    'fork': [
        'fork',
        'eating fork',
        'dining fork',
        'silverware fork'
    ],
    'spoon': [
        'spoon',
        'eating spoon',
        'soup spoon',
        'silverware spoon'
    ],
    'knife': [
        'knife',
        'eating knife',
        'dining knife',
        'silverware knife'
    ],
    'plate': [
        'plate',
        'dinner plate',
        'food plate',
        'eating plate'
    ],
    'gangsign': [
        'gang sign',
        'gang symbol',
        'gang hand sign'
    ],
    'thumbsup': [
        'thumbs up',
        'thumbs up gesture',
        'positive hand sign',
        'approval gesture'
    ],
    'thumbsdown': [
        'thumbs down',
        'thumbs down gesture',
        'negative hand sign',
        'disapproval gesture'
    ],
    'okay': [
        'okay sign',
        'ok hand gesture',
        'ok symbol',
        'ok hand sign'
    ],
    'peace': [
        'peace sign',
        'peace gesture',
        'peace hand sign',
        'victory peace sign'
    ],
    'fuckoff': [
        'middle finger',
        'rude gesture',
        'offensive hand sign',
        'angry gesture'
    ]
}

# Function to get smoothed prediction
def get_smoothed_prediction(prediction_buffer):
    if not prediction_buffer:
        return None
    # Average similarities over the buffer
    avg_similarities = np.mean(prediction_buffer, axis=0)
    return avg_similarities

def run_zeroshot_inference():
    # Initialize CLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device)
    model.eval()

    # Encode text prompts
    text_features = {}
    for class_name, prompts in CLASS_PROMPTS.items():
        # Encode all prompts for this class
        prompt_features = []
        for prompt in prompts:
            text_tokens = tokenizer([prompt]).to(device)
            with torch.no_grad():
                text_feature = model.encode_text(text_tokens)
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                prompt_features.append(text_feature)
        text_features[class_name] = prompt_features

    # Temporal smoothing buffer
    buffer_size = 15  # Number of frames to consider for smoothing
    prediction_buffer = deque(maxlen=buffer_size)

    try:
        # Try to open DroidCam
        for device_id in [1, 2]:
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increased resolution
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print(f"Successfully connected to DroidCam on device {device_id}")
                break
        else:
            print("Error: Could not connect to DroidCam. Make sure it's running and connected.")
            return

        print("DroidCam started. Press 'q' to quit.")

        # Create a resizable window
        cv2.namedWindow("CLIP Zero-shot Classification", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CLIP Zero-shot Classification", 1280, 720)

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
                    for class_name, prompt_features in text_features.items():
                        # Calculate similarity with each prompt and average
                        class_similarities = []
                        for text_feature in prompt_features:
                            similarity = torch.nn.functional.cosine_similarity(image_features, text_feature)
                            class_similarities.append(similarity.item())
                        similarities[class_name] = np.mean(class_similarities)
                    
                    # Convert to array and add to buffer
                    similarity_array = np.array(list(similarities.values()))
                    prediction_buffer.append(similarity_array)

                # Get smoothed prediction
                smoothed_similarities = get_smoothed_prediction(prediction_buffer)
                if smoothed_similarities is not None:
                    predicted_idx = np.argmax(smoothed_similarities)  # Highest similarity is best match
                    predicted_label = list(CLASS_PROMPTS.keys())[predicted_idx]

                    # Create a copy of the frame for display
                    display_frame = frame.copy()
                    
                    # Add a semi-transparent overlay for text
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (300, display_frame.shape[0]), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

                    # Display predictions in two columns
                    y_offset = 30
                    for i, (class_name, similarity) in enumerate(zip(CLASS_PROMPTS.keys(), smoothed_similarities)):
                        similarity_text = f"{class_name}: {similarity:.3f}"
                        color = (0, 255, 0) if i == predicted_idx else (255, 255, 255)
                        cv2.putText(display_frame, similarity_text, (10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Smaller text
                        y_offset += 20  # Reduced spacing

                    cv2.imshow("CLIP Zero-shot Classification", display_frame)
                
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
    run_zeroshot_inference() 