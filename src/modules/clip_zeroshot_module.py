import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import time
from datetime import timedelta

# Add the src directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.base_module import BaseModule
from model_clipZeroShot import CLIPZeroShotModel

class ActivityTracker:
    """Activity tracking and XP system for CLIP zero-shot module."""
    
    def __init__(self):
        self.current_activity = None
        self.start_time = None
        self.confidence = 0.0
        self.activity_history = []
        self.base_xp = 0  # Base XP from completed activities
        self.xp_per_activity = 60  # XP gained per minute of activity
        
        # Activity switching parameters
        self.min_activity_duration = 3.0  # Minimum seconds before allowing activity change
        self.confidence_threshold = 0.2  # Minimum confidence to consider an activity
        self.hysteresis_threshold = 0.0   # How much more confident new activity needs to be
        self.switch_confirmation_time = 3.0  # How long new activity must be detected before switching
        self.last_switch_time = 0
        
        # Switch confirmation tracking
        self.pending_activity = None
        self.pending_confidence = 0.0
        self.pending_start_time = None
        
    def update_activity(self, activity, confidence):
        current_time = time.time()
        
        # First, check if we should switch activities
        should_switch = False
        
        # Case 1: No current activity, and new activity meets confidence threshold
        if self.current_activity is None and confidence >= self.confidence_threshold:
            should_switch = True
            
        # Case 2: Have current activity, and new activity is different and better
        elif (self.current_activity is not None and 
              activity != self.current_activity and
              confidence >= self.confidence_threshold and
              current_time - self.last_switch_time >= self.min_activity_duration):
            # If hysteresis is enabled, check if new activity is significantly better
            if self.hysteresis_threshold > 0:
                if confidence >= self.confidence + self.hysteresis_threshold:
                    should_switch = True
            else:
                should_switch = True
        
        # Handle switch confirmation
        if should_switch:
            if self.pending_activity != activity:
                # New pending activity, start tracking it
                self.pending_activity = activity
                self.pending_confidence = confidence
                self.pending_start_time = current_time
            elif current_time - self.pending_start_time >= self.switch_confirmation_time:
                # Pending activity has been consistent long enough, perform the switch
                if self.current_activity is not None:
                    # Save previous activity duration and add to base XP
                    duration = time.time() - self.start_time
                    xp_gained = int(duration / 60 * self.xp_per_activity)
                    self.activity_history.append({
                        'activity': self.current_activity,
                        'duration': duration,
                        'xp_gained': xp_gained
                    })
                    self.base_xp += xp_gained
                
                self.current_activity = activity
                self.start_time = time.time()
                self.last_switch_time = current_time
                self.pending_activity = None  # Reset pending activity
        else:
            # Reset pending activity if conditions are no longer met
            self.pending_activity = None
            
        # Always update confidence for current activity
        self.confidence = confidence
    
    def get_current_duration(self):
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def get_current_xp(self):
        # Calculate XP from current activity
        current_xp = self.base_xp
        if self.start_time is not None:
            current_duration = self.get_current_duration()
            current_xp += int(current_duration / 60 * self.xp_per_activity)
        return current_xp
    
    def get_xp_progress(self):
        return self.get_current_xp() % 100  # Show progress to next level (0-99)

    def get_activity_summary(self):
        summary = []
        for activity in self.activity_history:
            summary.append(f"{activity['activity']}: {timedelta(seconds=int(activity['duration']))} (XP: +{activity['xp_gained']})")
        summary.append(f"Total XP: {self.get_current_xp()}")
        return summary

class CLIPZeroShotModule(BaseModule):
    """CLIP zero-shot activity classification module."""
    
    def __init__(self, buffer_size=30, inference_interval=1):
        super().__init__()
        self.buffer_size = buffer_size
        self.clip_model = CLIPZeroShotModel(buffer_size=buffer_size, inference_interval=inference_interval)
        self.activity_tracker = ActivityTracker()
        self.last_activity = "None"
    
    def get_plugin_requirements(self):
        """Define plugin requirements for CLIP zero-shot module."""
        return {
            'camera': {
                'type': 'droidcam',  # Default camera type
                'config': {
                    'device_id': 1,
                    'width': 1280,
                    'height': 720
                }
            },
            'display': {
                'type': 'cv2',  # Default display type
                'config': {
                    'scale': 4,
                    'window_name': 'CLIP Zero-Shot Preview'
                }
            }
        }
    
    def set_camera_type(self, camera_type, **config):
        """Change camera type and configuration."""
        if camera_type not in ['droidcam', 'url', 'test']:
            raise ValueError(f"Unknown camera type: {camera_type}")
        
        # Update camera configuration
        self.plugin_configs['camera'] = {
            'type': camera_type,
            'config': config
        }
        
        # If camera plugin is already initialized, reinitialize it
        if 'camera' in self.plugins:
            old_camera = self.plugins['camera']
            old_camera.release()
            del self.plugins['camera']
            
            # Initialize new camera plugin
            self._initialize_plugin('camera', self.plugin_configs['camera'])
    
    def set_display_type(self, display_type, **config):
        """Change display type and configuration."""
        if display_type not in ['cv2', 'console', 'none']:
            raise ValueError(f"Unknown display type: {display_type}")
        
        # Update display configuration
        self.plugin_configs['display'] = {
            'type': display_type,
            'config': config
        }
        
        # If display plugin is already initialized, reinitialize it
        if 'display' in self.plugins:
            old_display = self.plugins['display']
            old_display.cleanup()
            del self.plugins['display']
            
            # Initialize new display plugin
            self._initialize_plugin('display', self.plugin_configs['display'])
        
    def process_frame(self, frame: np.ndarray) -> Image.Image:
        """
        Process a camera frame using CLIP zero-shot classification.
        
        Args:
            frame: Camera frame as numpy array (BGR format from OpenCV)
            
        Returns:
            PIL Image object with activity display for ESP32 HUD
        """
        # Get prediction from CLIP model
        prediction = self.clip_model.predict(frame)
        
        # Update activity tracker
        self.activity_tracker.update_activity(prediction.label, prediction.confidence)
        self.last_activity = prediction.label
        
        # Create bitmap for ESP32 HUD
        bitmap = self._generate_bitmap()
        
        return bitmap
    
    def _generate_bitmap(self):
        """Generate a PIL Image bitmap for ESP32 HUD.
        
        Returns:
            PIL.Image: 128x64 monochrome image ready for ESP32 OLED
        """
        # Create a new image with white background (will be inverted for OLED)
        image = Image.new('L', (128, 64), 255)
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            font_small = ImageFont.truetype("arial.ttf", 8)
            font_medium = ImageFont.truetype("arial.ttf", 10)
            font_large = ImageFont.truetype("arial.ttf", 12)
        except:
            font_small = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_large = ImageFont.load_default()
        
        # Get activity data
        current_activity = self.activity_tracker.current_activity or "No Activity"
        confidence = self.activity_tracker.confidence
        duration = self.activity_tracker.get_current_duration()
        current_xp = self.activity_tracker.get_current_xp()
        current_level = current_xp // 100
        xp_progress = current_xp % 100
        
        # Draw activity name (top left)
        activity_text = f"{current_activity[:10]}"  # Truncate if too long
        draw.text((2, 2), activity_text, fill=0, font=font_medium)
        
        # Draw confidence (top right)
        confidence_text = f"{confidence:.1%}"
        confidence_bbox = draw.textbbox((0, 0), confidence_text, font=font_small)
        confidence_width = confidence_bbox[2] - confidence_bbox[0]
        draw.text((128 - confidence_width - 2, 2), confidence_text, fill=0, font=font_small)
        
        # Draw timer (center top)
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        timer_text = f"{minutes:02d}:{seconds:02d}"
        timer_bbox = draw.textbbox((0, 0), timer_text, font=font_large)
        timer_width = timer_bbox[2] - timer_bbox[0]
        timer_x = (128 - timer_width) // 2
        draw.text((timer_x, 15), timer_text, fill=0, font=font_large)
        
        # Draw XP bar (middle)
        bar_y = 35
        bar_height = 8
        bar_width = 128 - 4
        
        # XP bar background
        draw.rectangle([(2, bar_y), (128 - 2, bar_y + bar_height)], outline=0, width=1)
        
        # XP progress fill
        progress_width = int((xp_progress / 100) * (bar_width - 2))
        if progress_width > 0:
            draw.rectangle([(3, bar_y + 1), (3 + progress_width, bar_y + bar_height - 1)], fill=0)
        
        # Draw XP and level info (bottom)
        xp_text = f"XP: {current_xp}"
        level_text = f"Lv.{current_level}"
        
        draw.text((2, bar_y + bar_height + 5), xp_text, fill=0, font=font_small)
        
        level_bbox = draw.textbbox((0, 0), level_text, font=font_small)
        level_width = level_bbox[2] - level_bbox[0]
        draw.text((128 - level_width - 2, bar_y + bar_height + 5), level_text, fill=0, font=font_small)
        
        # Draw pending activity if there is one
        if self.activity_tracker.pending_activity is not None:
            pending_time = time.time() - self.activity_tracker.pending_start_time
            pending_text = f"-> {self.activity_tracker.pending_activity[:8]} ({pending_time:.1f}s)"
            draw.text((2, 64 - 12), pending_text, fill=0, font=font_small)
        
        # Invert the image for OLED (black background, white text)
        image = Image.eval(image, lambda x: 255 - x)
        
        return image
    
    def cleanup(self):
        """Clean up resources and print activity summary."""
        self.clip_model.cleanup()
        print("\nActivity Summary:")
        for line in self.activity_tracker.get_activity_summary():
            print(line) 