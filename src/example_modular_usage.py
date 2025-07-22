#!/usr/bin/env python3
"""
Example showing how to use the modular HUD inference system.
Demonstrates different plugin configurations and module usage.
"""

import sys
import os
import time

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.clip_zeroshot_module import CLIPZeroShotModule

def example_basic_usage():
    """Example 1: Basic usage with default plugins."""
    print("=== Example 1: Basic Usage ===")
    
    # Create module (uses default droidcam + cv2 display)
    module = CLIPZeroShotModule()
    
    # Initialize plugins
    if not module.initialize_plugins():
        print("Failed to initialize plugins")
        return
    
    # Run for a few seconds
    camera_plugin = module.get_plugin('camera')
    display_plugin = module.get_plugin('display')
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < 5.0:  # Run for 5 seconds
            ret, frame = camera_plugin.read()
            if ret and frame is not None:
                bitmap = module.process_frame(frame)
                if display_plugin:
                    display_plugin.show_frame(bitmap)
                frame_count += 1
                time.sleep(0.033)  # ~30 FPS
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    print(f"Processed {frame_count} frames in 5 seconds")
    module.cleanup()

def example_test_mode():
    """Example 2: Test mode with dummy camera."""
    print("\n=== Example 2: Test Mode ===")
    
    # Create module
    module = CLIPZeroShotModule()
    
    # Change to test camera
    module.set_camera_type('test', width=640, height=480)
    
    # Change to console display
    module.set_display_type('console', scale=0.3)
    
    # Initialize plugins
    if not module.initialize_plugins():
        print("Failed to initialize plugins")
        return
    
    # Run for a few seconds
    camera_plugin = module.get_plugin('camera')
    display_plugin = module.get_plugin('display')
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < 3.0:  # Run for 3 seconds
            ret, frame = camera_plugin.read()
            if ret and frame is not None:
                bitmap = module.process_frame(frame)
                if display_plugin:
                    display_plugin.show_frame(bitmap)
                frame_count += 1
                time.sleep(0.1)  # Slower for console display
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    print(f"Processed {frame_count} frames in 3 seconds")
    module.cleanup()

def example_url_camera():
    """Example 3: URL camera with headless mode."""
    print("\n=== Example 3: URL Camera (Headless) ===")
    
    # Create module
    module = CLIPZeroShotModule()
    
    # Change to URL camera
    module.set_camera_type('url', url="http://192.168.0.25:81/stream")
    
    # Change to no display (headless)
    module.set_display_type('none')
    
    # Initialize plugins
    if not module.initialize_plugins():
        print("Failed to initialize plugins")
        return
    
    # Run for a few seconds
    camera_plugin = module.get_plugin('camera')
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < 3.0:  # Run for 3 seconds
            ret, frame = camera_plugin.read()
            if ret and frame is not None:
                bitmap = module.process_frame(frame)
                # In headless mode, we could save the bitmap or send it somewhere
                frame_count += 1
                time.sleep(0.033)  # ~30 FPS
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    print(f"Processed {frame_count} frames in 3 seconds (headless mode)")
    module.cleanup()

def example_dynamic_plugin_switching():
    """Example 4: Dynamically switching plugins."""
    print("\n=== Example 4: Dynamic Plugin Switching ===")
    
    # Create module
    module = CLIPZeroShotModule()
    
    # Start with test camera and console display
    module.set_camera_type('test', width=320, height=240)
    module.set_display_type('console', scale=0.5)
    
    # Initialize plugins
    if not module.initialize_plugins():
        print("Failed to initialize plugins")
        return
    
    camera_plugin = module.get_plugin('camera')
    display_plugin = module.get_plugin('display')
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < 6.0:  # Run for 6 seconds
            ret, frame = camera_plugin.read()
            if ret and frame is not None:
                bitmap = module.process_frame(frame)
                if display_plugin:
                    display_plugin.show_frame(bitmap)
                frame_count += 1
                
                # Switch display type after 3 seconds
                if frame_count == 90:  # ~3 seconds at 30 FPS
                    print("\nSwitching to CV2 display...")
                    module.set_display_type('cv2', scale=6, window_name='Dynamic Switch Demo')
                    display_plugin = module.get_plugin('display')
                
                time.sleep(0.033)
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    print(f"Processed {frame_count} frames in 6 seconds")
    module.cleanup()

def main():
    """Run all examples."""
    print("Modular HUD Inference System - Examples")
    print("=" * 50)
    
    try:
        # Example 1: Basic usage (requires DroidCam)
        example_basic_usage()
        
        # Example 2: Test mode (no hardware required)
        example_test_mode()
        
        # Example 3: URL camera (requires ESP32-CAM)
        example_url_camera()
        
        # Example 4: Dynamic switching
        example_dynamic_plugin_switching()
        
    except Exception as e:
        print(f"Error running examples: {e}")
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    main() 