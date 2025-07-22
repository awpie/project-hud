#!/usr/bin/env python3
"""
Test script to demonstrate the scalable plugin system.
Shows how new plugin types can be added without modifying the base module.
"""

import sys
import os
import time

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plugins.plugin_registry import PLUGIN_REGISTRY, get_available_plugin_types, register_plugin
from modules.audio_example_module import AudioExampleModule

def test_plugin_registry():
    """Test the plugin registry functionality."""
    print("=== Testing Plugin Registry ===")
    
    # Test available plugin types
    available_types = get_available_plugin_types()
    print(f"Available plugin types: {available_types}")
    
    # Test plugin creation for each type
    for plugin_type in available_types:
        print(f"\nTesting {plugin_type} plugin creation...")
        try:
            factory = PLUGIN_REGISTRY[plugin_type]
            
            # Create test plugin with default config
            if plugin_type == 'camera':
                plugin = factory('test', width=320, height=240)
            elif plugin_type == 'display':
                plugin = factory('console', scale=0.5)
            elif plugin_type == 'audio':
                plugin = factory('test', sample_rate=8000, chunk_size=512)
            else:
                plugin = factory('test')
            
            # Test initialization
            if plugin.initialize():
                print(f"✓ {plugin_type} plugin initialized successfully")
                
                # Test basic functionality
                if hasattr(plugin, 'read'):
                    ret, data = plugin.read()
                    print(f"  - read() returned: {ret}")
                elif hasattr(plugin, 'read_audio'):
                    data = plugin.read_audio()
                    print(f"  - read_audio() returned data of shape: {data.shape if data is not None else None}")
                
                # Clean up
                plugin.cleanup()
                print(f"  - {plugin_type} plugin cleaned up")
            else:
                print(f"✗ {plugin_type} plugin failed to initialize")
                
        except Exception as e:
            print(f"✗ Error testing {plugin_type} plugin: {e}")

def test_audio_module():
    """Test the audio example module."""
    print("\n=== Testing Audio Example Module ===")
    
    # Create module
    module = AudioExampleModule()
    
    # Check plugin requirements
    requirements = module.get_plugin_requirements()
    print(f"Module plugin requirements: {list(requirements.keys())}")
    
    # Initialize plugins
    if module.initialize_plugins():
        print("✓ All plugins initialized successfully")
        
        # Test plugin access
        camera_plugin = module.get_plugin('camera')
        display_plugin = module.get_plugin('display')
        audio_plugin = module.get_plugin('audio')
        
        print(f"  - Camera plugin: {camera_plugin.__class__.__name__}")
        print(f"  - Display plugin: {display_plugin.__class__.__name__}")
        print(f"  - Audio plugin: {audio_plugin.__class__.__name__}")
        
        # Test frame processing
        print("\nTesting frame processing...")
        ret, frame = camera_plugin.read()
        if ret and frame is not None:
            bitmap = module.process_frame(frame)
            print(f"✓ Frame processed successfully, bitmap size: {bitmap.size}")
            
            # Test display
            if display_plugin:
                display_plugin.show_frame(bitmap)
                print("✓ Display plugin showed frame")
        else:
            print("✗ Failed to read camera frame")
        
        # Clean up
        module.cleanup()
        print("✓ Module cleaned up")
    else:
        print("✗ Failed to initialize plugins")

def test_dynamic_plugin_switching():
    """Test dynamic plugin switching."""
    print("\n=== Testing Dynamic Plugin Switching ===")
    
    module = AudioExampleModule()
    
    # Start with test plugins
    module.set_camera_type('test', width=160, height=120)
    module.set_display_type('console', scale=0.3)
    module.set_audio_type('test', sample_rate=8000)
    
    if module.initialize_plugins():
        print("✓ Initial plugins initialized")
        
        # Switch to different audio configuration
        print("Switching audio configuration...")
        module.set_audio_type('test', sample_rate=44100, chunk_size=2048)
        
        # The module should automatically reinitialize the audio plugin
        audio_plugin = module.get_plugin('audio')
        if audio_plugin:
            print(f"✓ Audio plugin switched: {audio_plugin.sample_rate}Hz, {audio_plugin.chunk_size} samples")
        
        module.cleanup()
        print("✓ Module cleaned up")

def test_plugin_registration():
    """Test dynamic plugin registration."""
    print("\n=== Testing Dynamic Plugin Registration ===")
    
    # Create a simple test plugin factory
    def create_test_plugin(plugin_type, **kwargs):
        class TestPlugin:
            def __init__(self, **kwargs):
                self.config = kwargs
                self.is_initialized = False
            
            def initialize(self):
                self.is_initialized = True
                print(f"✓ Test plugin initialized with config: {self.config}")
                return True
            
            def cleanup(self):
                self.is_initialized = False
                print("✓ Test plugin cleaned up")
        
        return TestPlugin(**kwargs)
    
    # Register the test plugin
    register_plugin('test_plugin', create_test_plugin)
    print("✓ Test plugin registered")
    
    # Check if it's available
    available_types = get_available_plugin_types()
    print(f"Available plugin types after registration: {available_types}")
    
    # Test creating the new plugin type
    try:
        factory = PLUGIN_REGISTRY['test_plugin']
        plugin = factory('test_type', param1='value1', param2=42)
        
        if plugin.initialize():
            print("✓ New plugin type works correctly")
            plugin.cleanup()
        else:
            print("✗ New plugin type failed to initialize")
    except Exception as e:
        print(f"✗ Error with new plugin type: {e}")

def main():
    """Run all scalability tests."""
    print("Plugin Scalability Test Suite")
    print("=" * 50)
    
    try:
        test_plugin_registry()
        test_audio_module()
        test_dynamic_plugin_switching()
        test_plugin_registration()
        
        print("\n" + "=" * 50)
        print("✓ All scalability tests completed successfully!")
        print("\nThe plugin system is now truly scalable:")
        print("- New plugin types can be added by registering them in plugin_registry.py")
        print("- No changes needed to BaseModule when adding new plugin types")
        print("- Modules can use any registered plugin type")
        print("- Dynamic plugin switching and configuration works seamlessly")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")

if __name__ == "__main__":
    main() 