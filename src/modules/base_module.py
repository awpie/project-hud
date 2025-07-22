from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
from typing import Dict, List, Any, Optional

class BaseModule(ABC):
    """Base class for all HUD inference modules."""
    
    def __init__(self):
        """Initialize the module. Override to set up models, config, etc."""
        self.plugins = {}
        self.plugin_configs = self.get_plugin_requirements()
    
    def get_plugin_requirements(self) -> Dict[str, Dict[str, Any]]:
        """
        Define plugin requirements for this module.
        
        Returns:
            Dict mapping plugin type to configuration dict.
            Example:
            {
                'camera': {'type': 'droidcam', 'config': {'device_id': 1}},
                'display': {'type': 'cv2', 'config': {'scale': 4}},
                'audio': {'type': 'microphone', 'config': {'sample_rate': 16000}}
            }
        """
        return {}
    
    def initialize_plugins(self) -> bool:
        """
        Initialize all required plugins for this module.
        
        Returns:
            bool: True if all plugins initialized successfully
        """
        try:
            for plugin_type, config in self.plugin_configs.items():
                if not self._initialize_plugin(plugin_type, config):
                    print(f"Failed to initialize {plugin_type} plugin")
                    return False
            return True
        except Exception as e:
            print(f"Error initializing plugins: {e}")
            return False
    
    def _initialize_plugin(self, plugin_type: str, config: Dict[str, Any]) -> bool:
        """Initialize a specific plugin type using plugin registry."""
        try:
            # Import the plugin registry
            from plugins.plugin_registry import PLUGIN_REGISTRY
            
            if plugin_type not in PLUGIN_REGISTRY:
                print(f"Unknown plugin type: {plugin_type}")
                return False
            
            # Get the plugin factory function from registry
            plugin_factory = PLUGIN_REGISTRY[plugin_type]
            
            # Create and initialize the plugin
            plugin = plugin_factory(config['type'], **config.get('config', {}))
            if plugin.initialize():
                self.plugins[plugin_type] = plugin
                return True
            return False
                
        except Exception as e:
            print(f"Error initializing {plugin_type} plugin: {e}")
            return False
    
    def get_plugin(self, plugin_type: str):
        """Get a specific plugin by type."""
        return self.plugins.get(plugin_type)
    
    def set_plugin_type(self, plugin_type: str, plugin_subtype: str, **config):
        """Change any plugin type and configuration."""
        # Update plugin configuration
        self.plugin_configs[plugin_type] = {
            'type': plugin_subtype,
            'config': config
        }
        
        # If plugin is already initialized, reinitialize it
        if plugin_type in self.plugins:
            old_plugin = self.plugins[plugin_type]
            
            # Use appropriate cleanup method
            if hasattr(old_plugin, 'cleanup'):
                old_plugin.cleanup()
            elif hasattr(old_plugin, 'release'):
                old_plugin.release()
            
            del self.plugins[plugin_type]
            
            # Initialize new plugin
            self._initialize_plugin(plugin_type, self.plugin_configs[plugin_type])
    
    # Convenience methods for common plugin types (optional)
    def set_camera_type(self, camera_type, **config):
        """Change camera type and configuration."""
        self.set_plugin_type('camera', camera_type, **config)
    
    def set_display_type(self, display_type, **config):
        """Change display type and configuration."""
        self.set_plugin_type('display', display_type, **config)
    
    def set_audio_type(self, audio_type, **config):
        """Change audio type and configuration."""
        self.set_plugin_type('audio', audio_type, **config)
    
    def cleanup_plugins(self):
        """Clean up all plugins."""
        for plugin_type, plugin in self.plugins.items():
            try:
                plugin.cleanup()
            except Exception as e:
                print(f"Error cleaning up {plugin_type} plugin: {e}")
        self.plugins.clear()
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Image.Image:
        """
        Process a camera frame and return a bitmap for ESP32 HUD.
        
        Args:
            frame: Camera frame as numpy array (BGR format from OpenCV)
            
        Returns:
            PIL Image object ready to be sent to ESP32
        """
        pass
    
    def cleanup(self):
        """Clean up resources when module is done. Override if needed."""
        self.cleanup_plugins() 