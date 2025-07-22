#!/usr/bin/env python3
"""
Example showing how to use the CLIP zero-shot module independently.
This demonstrates the self-contained nature of the module.
"""

import sys
import os
import numpy as np
from PIL import Image

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.clip_zeroshot_module import CLIPZeroShotModule

def create_test_frame():
    """Create a simple test frame for demonstration."""
    # Create a 640x480 test image with some patterns
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate different activities
    # Red rectangle (could be interpreted as "eating" or "cooking")
    frame[100:200, 100:300] = [0, 0, 255]  # Red in BGR
    
    # Green rectangle (could be interpreted as "nature" or "outdoor")
    frame[250:350, 400:600] = [0, 255, 0]  # Green in BGR
    
    # Blue rectangle (could be interpreted as "coding" or "computer work")
    frame[350:450, 50:250] = [255, 0, 0]   # Blue in BGR
    
    return frame

def main():
    """Demonstrate independent module usage."""
    print("=== CLIP Zero-Shot Module Independent Usage Example ===\n")
    
    # Create the module
    print("1. Creating CLIP Zero-Shot Module...")
    module = CLIPZeroShotModule(buffer_size=5)  # Small buffer for faster processing
    print("✓ Module created successfully")
    
    # Create a test frame
    print("\n2. Creating test frame...")
    test_frame = create_test_frame()
    print("✓ Test frame created (640x480)")
    
    # Process multiple frames to see activity tracking in action
    print("\n3. Processing frames to demonstrate activity tracking...")
    
    for i in range(10):
        print(f"\n   Frame {i+1}:")
        
        # Process the frame
        bitmap = module.process_frame(test_frame)
        
        # Get current activity info
        current_activity = module.activity_tracker.current_activity
        confidence = module.activity_tracker.confidence
        duration = module.activity_tracker.get_current_duration()
        xp = module.activity_tracker.get_current_xp()
        
        print(f"   - Activity: {current_activity}")
        print(f"   - Confidence: {confidence:.3f}")
        print(f"   - Duration: {duration:.1f}s")
        print(f"   - XP: {xp}")
        
        # Show bitmap info
        print(f"   - Bitmap size: {bitmap.size}")
        
        # Small delay to simulate real-time processing
        import time
        time.sleep(0.5)
    
    # Show final summary
    print("\n4. Final Activity Summary:")
    module.cleanup()
    
    print("\n=== Example Complete ===")
    print("This demonstrates how the module is completely self-contained!")
    print("It handles:")
    print("- CLIP inference")
    print("- Activity tracking and switching")
    print("- XP system")
    print("- Bitmap generation")
    print("- All without external dependencies")

if __name__ == "__main__":
    main() 