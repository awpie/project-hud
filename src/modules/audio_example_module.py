#!/usr/bin/env python3
"""
Example module that demonstrates the use of audio plugins.
This shows how the scalable plugin system works with new plugin types.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import time

# Add the src directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.base_module import BaseModule

class AudioExampleModule(BaseModule):
    """Example module that uses camera and audio plugins."""
    
    def __init__(self):
        super().__init__()
        self.audio_level = 0.0
        self.frame_count = 0
    
    def get_plugin_requirements(self):
        """Define plugin requirements for this module."""
        return {
            'camera': {
                'type': 'test',  # Use test camera for demo
                'config': {
                    'width': 640,
                    'height': 480
                }
            },
            'display': {
                'type': 'cv2',
                'config': {
                    'scale': 4,
                    'window_name': 'Audio Example Module'
                }
            },
            'audio': {
                'type': 'test',  # Use test audio for demo
                'config': {
                    'sample_rate': 16000,
                    'chunk_size': 1024
                }
            }
        }
    
    def process_frame(self, frame: np.ndarray) -> Image.Image:
        """
        Process a camera frame and audio data.
        
        Args:
            frame: Camera frame as numpy array (BGR format from OpenCV)
            
        Returns:
            PIL Image object with audio visualization for ESP32 HUD
        """
        self.frame_count += 1
        
        # Get audio data from audio plugin
        audio_plugin = self.get_plugin('audio')
        if audio_plugin and audio_plugin.is_active():
            audio_data = audio_plugin.read_audio()
            if audio_data is not None:
                # Calculate audio level (RMS)
                self.audio_level = np.sqrt(np.mean(audio_data**2))
        
        # Create bitmap for ESP32 HUD
        bitmap = self._generate_bitmap()
        
        return bitmap
    
    def _generate_bitmap(self):
        """Generate a PIL Image bitmap showing audio visualization."""
        # Create a new image with white background (will be inverted for OLED)
        image = Image.new('L', (128, 64), 255)
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            font_small = ImageFont.truetype("arial.ttf", 8)
            font_medium = ImageFont.truetype("arial.ttf", 10)
        except:
            font_small = ImageFont.load_default()
            font_medium = ImageFont.load_default()
        
        # Draw title
        title = "Audio Demo"
        draw.text((2, 2), title, fill=0, font=font_medium)
        
        # Draw frame count
        frame_text = f"Frame: {self.frame_count}"
        draw.text((2, 15), frame_text, fill=0, font=font_small)
        
        # Draw audio level
        audio_text = f"Audio: {self.audio_level:.3f}"
        draw.text((2, 25), audio_text, fill=0, font=font_small)
        
        # Draw audio level bar
        bar_y = 40
        bar_height = 8
        bar_width = 128 - 4
        
        # Audio bar background
        draw.rectangle([(2, bar_y), (128 - 2, bar_y + bar_height)], outline=0, width=1)
        
        # Audio level fill (normalize to 0-1, then scale to bar width)
        normalized_level = min(self.audio_level * 10, 1.0)  # Scale up for visibility
        level_width = int(normalized_level * (bar_width - 2))
        if level_width > 0:
            draw.rectangle([(3, bar_y + 1), (3 + level_width, bar_y + bar_height - 1)], fill=0)
        
        # Draw status
        status = "Active" if self.get_plugin('audio') and self.get_plugin('audio').is_active() else "No Audio"
        draw.text((2, bar_y + bar_height + 5), status, fill=0, font=font_small)
        
        # Invert the image for OLED (black background, white text)
        image = Image.eval(image, lambda x: 255 - x)
        
        return image
    
    def cleanup(self):
        """Clean up resources and print summary."""
        super().cleanup()
        print(f"\nAudio Example Module Summary:")
        print(f"- Total frames processed: {self.frame_count}")
        print(f"- Final audio level: {self.audio_level:.3f}") 