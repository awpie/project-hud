#!/usr/bin/env python3
"""
Test script for the new modular architecture.
This script tests the basic functionality without requiring a camera.
"""

import sys
import os
import numpy as np
from PIL import Image

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_base_module():
    """Test the base module interface."""
    print("Testing BaseModule interface...")
    
    from modules.base_module import BaseModule
    
    # Create a simple test module
    class TestModule(BaseModule):
        def process_frame(self, frame):
            # Create a simple test bitmap
            image = Image.new('L', (128, 64), 255)
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Draw some test text
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), "Test Module", fill=0, font=font)
            draw.text((10, 30), "Working!", fill=0, font=font)
            
            # Invert for OLED
            image = Image.eval(image, lambda x: 255 - x)
            return image
    
    # Test the module
    module = TestModule()
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy frame
    bitmap = module.process_frame(test_frame)
    
    print(f"✓ Test module created bitmap: {bitmap.size}")
    return True

def test_clip_module():
    """Test the CLIP zero-shot module (without camera)."""
    print("Testing CLIP Zero-Shot Module...")
    
    try:
        from modules.clip_zeroshot_module import CLIPZeroShotModule
        
        # Create the module
        module = CLIPZeroShotModule()
        print("✓ CLIP Zero-Shot Module created successfully")
        
        # Test with a dummy frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[100:200, 100:200] = 255  # Add some white pixels
        
        try:
            bitmap = module.process_frame(test_frame)
            print(f"✓ CLIP module processed frame and created bitmap: {bitmap.size}")
            module.cleanup()
            return True
        except Exception as e:
            print(f"✗ CLIP module processing failed: {e}")
            return False
            
    except ImportError as e:
        print(f"✗ Could not import CLIP module: {e}")
        return False
    except Exception as e:
        print(f"✗ CLIP module creation failed: {e}")
        return False

def test_camera_plugins():
    """Test camera plugin creation."""
    print("Testing Camera Plugins...")
    
    try:
        from plugins.camera_plugins import create_camera_plugin
        
        # Test DroidCam plugin creation
        droidcam_plugin = create_camera_plugin('droidcam')
        print("✓ DroidCam plugin created")
        
        # Test URL plugin creation
        url_plugin = create_camera_plugin('url')
        print("✓ URL plugin created")
        
        return True
        
    except Exception as e:
        print(f"✗ Camera plugin test failed: {e}")
        return False

def test_esp32_client():
    """Test ESP32 client functions."""
    print("Testing ESP32 Client...")
    
    try:
        from execution.esp32_client import send_pil_image
        
        # Create a test image
        test_image = Image.new('L', (128, 64), 255)
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(test_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "Test", fill=0, font=font)
        test_image = Image.eval(test_image, lambda x: 255 - x)
        
        print("✓ Test image created")
        print("✓ send_pil_image function imported")
        
        return True
        
    except Exception as e:
        print(f"✗ ESP32 client test failed: {e}")
        return False

def test_display_plugins():
    """Test display plugin creation and functionality."""
    print("Testing Display Plugins...")
    
    try:
        from plugins.display_plugins import create_display_plugin
        
        # Test CV2 plugin creation
        cv2_plugin = create_display_plugin('cv2')
        print("✓ CV2 display plugin created")
        
        # Test console plugin creation
        console_plugin = create_display_plugin('console')
        print("✓ Console display plugin created")
        
        # Test no display plugin creation
        none_plugin = create_display_plugin('none')
        print("✓ No display plugin created")
        
        # Test plugin initialization
        if cv2_plugin.initialize():
            print("✓ CV2 plugin initialized")
            cv2_plugin.cleanup()
        else:
            print("⚠ CV2 plugin initialization failed (may be headless environment)")
        
        if console_plugin.initialize():
            print("✓ Console plugin initialized")
            console_plugin.cleanup()
        
        if none_plugin.initialize():
            print("✓ No display plugin initialized")
            none_plugin.cleanup()
        
        return True
        
    except Exception as e:
        print(f"✗ Display plugin test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Testing Modular Architecture ===\n")
    
    tests = [
        ("Base Module Interface", test_base_module),
        ("CLIP Zero-Shot Module", test_clip_module),
        ("Camera Plugins", test_camera_plugins),
        ("ESP32 Client", test_esp32_client),
        ("Display Plugins", test_display_plugins),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The modular architecture is working correctly.")
        print("\nYou can now run the main system with:")
        print("python src/run.py")
    else:
        print("⚠ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 