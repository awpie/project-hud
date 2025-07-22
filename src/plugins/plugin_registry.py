"""
Plugin Registry for the HUD inference system.

This registry maps plugin types to their factory functions, making the system
scalable and allowing new plugin types to be added without modifying the base module.
"""

# Import all plugin factory functions
from .camera_plugins import create_camera_plugin
from .display_plugins import create_display_plugin
from .audio_plugins import create_audio_plugin

# Plugin registry mapping plugin types to their factory functions
PLUGIN_REGISTRY = {
    'camera': create_camera_plugin,
    'display': create_display_plugin,
    'audio': create_audio_plugin,
    # Add new plugin types here as they're created:
    # 'radar': create_radar_plugin,
    # 'sensor': create_sensor_plugin,
}

def register_plugin(plugin_type: str, factory_function):
    """
    Register a new plugin type and its factory function.
    
    Args:
        plugin_type: The type name of the plugin (e.g., 'audio', 'radar')
        factory_function: The factory function that creates the plugin
    """
    PLUGIN_REGISTRY[plugin_type] = factory_function

def get_available_plugin_types():
    """Get a list of all available plugin types."""
    return list(PLUGIN_REGISTRY.keys())

def is_plugin_type_available(plugin_type: str) -> bool:
    """Check if a plugin type is available."""
    return plugin_type in PLUGIN_REGISTRY 