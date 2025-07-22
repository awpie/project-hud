#!/usr/bin/env python3
"""
Main runner for HUD inference modules.
Runs selected modules at 30 FPS and sends bitmap data to ESP32.
"""

import sys
import os
import time

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execution.esp32_client import send_pil_image, shutdown

def get_module_choice():
    """Get user's choice for inference module."""
    print("\nSelect inference module:")
    print("1. CLIP Zero-Shot Activity Classification")
    print("2. Audio Example Module (demonstrates plugin scalability)")
    # Add more modules here as they're created
    # print("3. Activity Classifier (trained model)")
    # print("4. Pose Detection")
    
    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice == '1':
                return 'clip_zeroshot'
            elif choice == '2':
                return 'audio_example'
            # elif choice == '3':
            #     return 'activity_classifier'
            # elif choice == '4':
            #     return 'pose_detection'
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
            return None
        except Exception as e:
            print(f"Error reading input: {e}")
            return None

def create_module(module_name):
    """Create and return the selected module."""
    if module_name == 'clip_zeroshot':
        from modules.clip_zeroshot_module import CLIPZeroShotModule
        return CLIPZeroShotModule()
    elif module_name == 'audio_example':
        from modules.audio_example_module import AudioExampleModule
        return AudioExampleModule()
    # elif module_name == 'activity_classifier':
    #     from modules.activity_classifier_module import ActivityClassifierModule
    #     return ActivityClassifierModule()
    # elif module_name == 'pose_detection':
    #     from modules.pose_detection_module import PoseDetectionModule
    #     return PoseDetectionModule()
    else:
        raise ValueError(f"Unknown module: {module_name}")

def run_module(module, fps=30):
    """
    Run the selected module at specified FPS.
    
    Args:
        module: The inference module to run
        fps: Target frames per second (default: 30)
    """
    frame_interval = 1.0 / fps
    last_frame_time = 0
    frame_count = 0
    start_time = time.time()
    
    print(f"Starting {module.__class__.__name__} at {fps} FPS...")
    print("Press 'q' to quit.")
    
    # Get plugins from module
    camera_plugin = module.get_plugin('camera')
    display_plugin = module.get_plugin('display')
    
    if not camera_plugin:
        print("Error: No camera plugin available")
        return
    
    try:
        while camera_plugin.is_opened() and (not display_plugin or display_plugin.is_active()):
            current_time = time.time()
            
            # Maintain target FPS
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            # Read frame from camera
            ret, frame = camera_plugin.read()
            if not ret or frame is None:
                print("Error: Could not read frame from camera.")
                continue
            
            # Process frame through module
            try:
                bitmap = module.process_frame(frame)
                
                # Send bitmap to ESP32
                send_pil_image(bitmap)
                
                # Show preview if display plugin exists
                if display_plugin:
                    display_plugin.show_frame(bitmap)
                
                frame_count += 1
                last_frame_time = current_time
                
                # Show FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed = current_time - start_time
                    actual_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"FPS: {actual_fps:.1f} | Frames: {frame_count}")
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\nModule execution interrupted by user.")
    except Exception as e:
        print(f"Error during module execution: {e}")
    finally:
        # Cleanup
        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nExecution completed:")
        print(f"- Total frames: {frame_count}")
        print(f"- Total time: {elapsed:.1f}s")
        print(f"- Average FPS: {actual_fps:.1f}")
        
        module.cleanup()
        shutdown()

def main():
    """Main function to run the HUD inference system."""
    print("=== HUD Inference System ===")
    
    # Get user choice
    module_name = get_module_choice()
    if module_name is None:
        print("No module selected. Exiting.")
        return
    
    try:
        # Create module
        print(f"Initializing {module_name} module...")
        module = create_module(module_name)
        
        # Initialize plugins (module handles its own plugin requirements)
        print("Initializing module plugins...")
        if not module.initialize_plugins():
            print("Failed to initialize module plugins. Exiting.")
            return
        
        # Run the module
        run_module(module)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("HUD inference system stopped.")

if __name__ == "__main__":
    main() 