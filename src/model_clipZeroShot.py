# CLIP Zero-shot Classification Model
import cv2
import torch
from PIL import Image
import numpy as np
from collections import deque
import open_clip
from prediction_result import PredictionResult

# Define text prompts for each class
CLASS_PROMPT = {
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
    'idle': [
        'view of random objects',
        'view of unmoving things',
        'furniture',
        'walls',
        'nothing interesting', 
        'view of a kitchen',
        'view of a bedroom',
        'view of a bathroom',
        'view of a living room',
        'view of a dining room',
        'view of a hallway',
        'view of a general scene'
    ]
}

def get_smoothed_prediction(prediction_buffer):
    """Get smoothed prediction using exponential weighting."""
    if not prediction_buffer:
        return None
    
    # Convert buffer to numpy array for easier manipulation
    buffer_array = np.array(prediction_buffer)
    
    # Apply exponential weighting (more recent predictions have higher weight)
    weights = np.exp(np.linspace(-1, 0, len(buffer_array)))
    weights = weights / weights.sum()
    
    # Calculate weighted average
    weighted_avg = np.average(buffer_array, weights=weights, axis=0)
    
    return weighted_avg

class CLIPZeroShotModel:
    """CLIP Zero-shot classification model with temporal smoothing."""
    
    def __init__(self, buffer_size=30, inference_interval=5):
        """Initialize the CLIP zero-shot model.
        
        Args:
            buffer_size (int): Size of the temporal smoothing buffer
            inference_interval (int): Run inference every N frames (default: 5)
        """
        self.buffer_size = buffer_size
        self.inference_interval = inference_interval
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.frame_count = 0
        self.last_prediction = None
        
        # Initialize CLIP model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Pre-encode text prompts
        self.text_features = self._encode_text_prompts()
        
        # Get class names in order
        self.class_names = list(CLASS_PROMPT.keys())
    
    def _encode_text_prompts(self):
        """Pre-encode all text prompts for efficiency."""
        text_features = {}
        for class_name, prompts in CLASS_PROMPT.items():
            # Encode all prompts for this class
            prompt_features = []
            for prompt in prompts:
                text_tokens = self.tokenizer([prompt]).to(self.device)
                with torch.no_grad():
                    text_feature = self.model.encode_text(text_tokens)
                    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                    prompt_features.append(text_feature)
            text_features[class_name] = prompt_features
        return text_features
    
    def predict(self, frame) -> PredictionResult:
        """Predict activity from a frame.
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            PredictionResult: Contains predicted label and confidence
        """
        self.frame_count += 1
        
        # Only run inference every N frames to improve performance
        if self.frame_count % self.inference_interval == 0:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Get CLIP features
            with torch.no_grad():
                image_features = self.model.encode_image(self.preprocess(pil_image).unsqueeze(0).to(self.device))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities with all text prompts
                similarities = {}
                for class_name, prompt_features in self.text_features.items():
                    # Calculate similarity with each prompt and average
                    class_similarities = []
                    for text_feature in prompt_features:
                        similarity = torch.nn.functional.cosine_similarity(image_features, text_feature)
                        class_similarities.append(similarity.item())
                    similarities[class_name] = np.mean(class_similarities)
                
                # Convert to array and add to buffer
                similarity_array = np.array(list(similarities.values()))
                self.prediction_buffer.append(similarity_array)
            
            # Get smoothed prediction
            smoothed_similarities = get_smoothed_prediction(self.prediction_buffer)
            if smoothed_similarities is not None:
                predicted_idx = np.argmax(smoothed_similarities)
                predicted_label = self.class_names[predicted_idx]
                confidence = smoothed_similarities[predicted_idx]
                
                self.last_prediction = PredictionResult(predicted_label, confidence)
        
        # Return cached prediction if available, otherwise default
        if self.last_prediction is not None:
            return self.last_prediction
        else:
            return PredictionResult("idle", 0.0)
    
    def get_all_predictions(self, frame):
        """Get all class predictions with their confidence scores.
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            dict: Dictionary mapping class names to confidence scores
        """
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Get CLIP features
        with torch.no_grad():
            image_features = self.model.encode_image(self.preprocess(pil_image).unsqueeze(0).to(self.device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities with all text prompts
            similarities = {}
            for class_name, prompt_features in self.text_features.items():
                # Calculate similarity with each prompt and average
                class_similarities = []
                for text_feature in prompt_features:
                    similarity = torch.nn.functional.cosine_similarity(image_features, text_feature)
                    class_similarities.append(similarity.item())
                similarities[class_name] = np.mean(class_similarities)
        
        return similarities
    
    def cleanup(self):
        """Clean up model resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 