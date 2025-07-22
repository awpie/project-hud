#!/usr/bin/env python3
"""
Example showing the generic plugin system in action.
Demonstrates how new plugin types can be used without any BaseModule changes.
"""

import sys
import os
import time

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.audio_example_module import AudioExampleModule

def demonstrate_generic_plugin_usage():
    """Demonstrate the generic plugin system."""
    print("=== Generic Plugin System Demo ===")
    
    # Create module
    module = AudioExampleModule()
    
    print("\n1. Using convenience methods (current approach):")
    module.set_camera_type('test', width=320, height=240)
    module.set_display_type('console', scale=0.3)
    module.set_audio_type('test', sample_rate=8000)
    
    print("\n2. Using generic method (new approach):")
    module.set_plugin_type('camera', 'test', width=160, height=120)
    module.set_plugin_type('display', 'console', scale=0.2)
    module.set_plugin_type('audio', 'test', sample_rate=44100, chunk_size=2048)
    
    print("\n3. Direct configuration (most generic):")
    module.plugin_configs = {
        'camera': {'type': 'test', 'config': {'width': 80, 'height': 60}},
        'display': {'type': 'none'},  # Headless mode
        'audio': {'type': 'test', 'config': {'sample_rate': 22050}}
    }
    
    # Initialize plugins
    if module.initialize_plugins():
        print("✓ All plugins initialized successfully")
        
        # Test the plugins
        camera_plugin = module.get_plugin('camera')
        display_plugin = module.get_plugin('display')
        audio_plugin = module.get_plugin('audio')
        
        print(f"  - Camera: {camera_plugin.__class__.__name__}")
        print(f"  - Display: {display_plugin.__class__.__name__}")
        print(f"  - Audio: {audio_plugin.__class__.__name__}")
        
        # Process a frame
        ret, frame = camera_plugin.read()
        if ret and frame is not None:
            bitmap = module.process_frame(frame)
            print(f"✓ Frame processed, bitmap size: {bitmap.size}")
        
        module.cleanup()
        print("✓ Module cleaned up")

def demonstrate_future_plugin_types():
    """Demonstrate how future plugin types would work."""
    print("\n=== Future Plugin Types Demo ===")
    
    # This shows how you could use future plugin types
    # without any changes to BaseModule
    
    module = AudioExampleModule()
    
    # Example: Using a hypothetical radar plugin
    # (This would work once radar_plugins.py is created and registered)
    print("\nExample: Using hypothetical radar plugin:")
    print("module.set_plugin_type('radar', 'test', range=100, frequency=24)")
    print("module.set_plugin_type('sensor', 'temperature', unit='celsius')")
    print("module.set_plugin_type('network', 'wifi', ssid='my_network')")
    
    # Example: Direct configuration for future plugins
    print("\nExample: Direct configuration for future plugins:")
    future_config = {
        'camera': {'type': 'test', 'config': {'width': 640, 'height': 480}},
        'display': {'type': 'cv2', 'config': {'scale': 4}},
        'audio': {'type': 'test', 'config': {'sample_rate': 16000}},
        # Future plugins (would work once implemented):
        # 'radar': {'type': 'test', 'config': {'range': 100}},
        # 'sensor': {'type': 'temperature', 'config': {'unit': 'celsius'}},
        # 'network': {'type': 'wifi', 'config': {'ssid': 'my_network'}},
    }
    
    print("module.plugin_configs = future_config")
    print("module.initialize_plugins()  # Automatically handles all plugin types!")
    
    # Clean up
    module.cleanup()

def main():
    """Run the generic plugin system demo."""
    print("Generic Plugin System Examples")
    print("=" * 50)
    
    try:
        demonstrate_generic_plugin_usage()
        demonstrate_future_plugin_types()
        
        print("\n" + "=" * 50)
        print("✓ Generic plugin system demo completed!")
        print("\nKey Points:")
        print("- set_plugin_type() works for ANY plugin type")
        print("- No BaseModule changes needed for new plugin types")
        print("- Convenience methods (set_camera_type, etc.) are optional")
        print("- Direct configuration is the most flexible approach")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")

if __name__ == "__main__":
    main() 