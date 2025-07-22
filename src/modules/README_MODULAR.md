# HUD Inference System - Modular Architecture

This document describes the new modular architecture for the HUD inference system.

## Overview

The system has been refactored from a centralized inference loop to a modular architecture where each inference pipeline is a self-contained module that processes camera frames and outputs bitmap data for the ESP32 HUD. **Modules now manage their own plugin requirements**, making the system more scalable and self-contained.

## Architecture

### Core Components

1. **Modules** (`src/modules/`)

   - Self-contained inference pipelines
   - Each module inherits from `BaseModule`
   - Implements `process_frame()` method that takes a camera frame and returns a PIL Image
   - **Manages its own plugin requirements** (camera, display, audio, etc.)

2. **Plugin Registry** (`src/plugins/plugin_registry.py`)

   - **Central registry mapping plugin types to factory functions**
   - **Scalable system - no need to modify BaseModule when adding new plugin types**
   - Automatic plugin discovery and initialization
   - **Dynamic plugin registration** for runtime extensibility

3. **Camera Plugins** (`src/plugins/`)

   - Reusable camera initialization logic
   - Supports DroidCam, ESP32-CAM streams, and test mode
   - Each plugin handles its own connection and frame reading
   - **Configurable via module requirements**

4. **Display Plugins** (`src/plugins/`)

   - Preview display options for debugging and monitoring
   - Supports CV2 window, console ASCII art, and headless mode
   - Each plugin handles its own display rendering
   - **Configurable via module requirements**

5. **Audio Plugins** (`src/plugins/`)

   - Audio input processing (microphone, test audio)
   - **Demonstrates the scalable plugin system**
   - Easy to extend with real audio hardware

6. **Plugin Management** (`BaseModule`)

   - Automatic plugin initialization based on module requirements
   - Dynamic plugin switching and configuration
   - **Truly scalable architecture** - new plugin types just need registration

7. **Main Runner** (`src/run.py`)
   - Simple launcher that asks user to select module
   - **Module automatically initializes its required plugins**
   - Runs selected module at 30 FPS
   - Handles ESP32 communication and preview display

### File Structure

```
src/
├── modules/
│   ├── __init__.py
│   ├── base_module.py              # Base class for all modules
│   └── clip_zeroshot_module.py     # CLIP zero-shot activity classification
├── plugins/
│   ├── __init__.py
│   ├── plugin_registry.py          # Central plugin registry
│   ├── camera_plugins.py           # Camera initialization plugins
│   ├── display_plugins.py          # Display preview plugins
│   └── audio_plugins.py            # Audio input plugins
├── run.py                          # Main launcher with ESP32 communication
├── preview.py                      # Preview launcher without ESP32
├── example_modular_usage.py        # Examples of modular usage
├── example_module_usage.py         # Simple module usage example
├── audio_example_module.py         # Example module using audio plugins
├── test_modular_architecture.py    # Test script
├── test_plugin_scalability.py      # Test script for plugin scalability
└── README_MODULAR.md               # This file
```

## Usage

### Running the System

1. **Test the architecture:**

   ```bash
   python src/test_modular_architecture.py
   ```

2. **Run the main system (with ESP32):**

   ```bash
   python src/run.py
   ```

3. **Run in preview mode (no ESP32):**

   ```bash
   python src/preview.py
   ```

4. **View examples:**

   ```bash
   python src/example_modular_usage.py
   ```

5. **Select your options:**
   - Choose inference module (CLIP zero-shot or Audio Example)
   - **The module automatically initializes its required plugins**
   - No need to manually select camera/display - modules handle this

### Creating New Modules

To create a new module:

1. **Create a new file** in `src/modules/` (e.g., `my_module.py`)

2. **Inherit from BaseModule and define plugin requirements:**

   ```python
   from modules.base_module import BaseModule
   import numpy as np
   from PIL import Image

   class MyModule(BaseModule):
       def __init__(self):
           super().__init__()
           # Initialize your models, config, etc.

       def get_plugin_requirements(self):
           """Define plugin requirements for this module."""
           return {
               'camera': {
                   'type': 'droidcam',
                   'config': {
                       'device_id': 1,
                       'width': 1280,
                       'height': 720
                   }
               },
               'display': {
                   'type': 'cv2',
                   'config': {
                       'scale': 4,
                       'window_name': 'My Module Preview'
                   }
               }
           }

       def process_frame(self, frame: np.ndarray) -> Image.Image:
           # Process the camera frame
           # Return a PIL Image (128x64 monochrome for ESP32 OLED)
           pass

       def cleanup(self):
           # Clean up resources (optional)
           super().cleanup()  # This cleans up plugins automatically
   ```

3. **Include your own display logic**: Each module should contain its own activity tracking, XP system, and bitmap generation logic. See `clip_zeroshot_module.py` for a complete example.

4. **Add to the runner** in `src/run.py`:

   ```python
   def create_module(module_name):
       if module_name == 'my_module':
           from modules.my_module import MyModule
           return MyModule()
       # ... existing code ...
   ```

5. **Add to the menu** in `src/run.py`:
   ```python
   def get_module_choice():
       print("1. CLIP Zero-Shot Activity Classification")
       print("2. My New Module")  # Add this line
       # ... update choice logic ...
   ```

## Module Interface

### BaseModule Class

```python
class BaseModule(ABC):
    def __init__(self):
        """Initialize the module. Override to set up models, config, etc."""
        self.plugins = {}
        self.plugin_configs = self.get_plugin_requirements()

    def get_plugin_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Define plugin requirements for this module. Override in subclasses."""
        return {}

    def initialize_plugins(self) -> bool:
        """Initialize all required plugins for this module."""
        # Automatically initializes plugins based on get_plugin_requirements()

    def get_plugin(self, plugin_type: str):
        """Get a specific plugin by type."""
        return self.plugins.get(plugin_type)

    def set_camera_type(self, camera_type, **config):
        """Change camera type and configuration."""
        # Dynamically switch camera plugins

    def set_display_type(self, display_type, **config):
        """Change display type and configuration."""
        # Dynamically switch display plugins

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
        self.cleanup_plugins()  # Automatically cleans up all plugins
```

### Camera Plugin Interface

```python
class CameraPlugin:
    def initialize(self):
        """Initialize the camera. Return True if successful."""
        pass

    def read(self):
        """Read a frame. Return (success, frame)."""
        pass

    def is_opened(self):
        """Check if camera is opened."""
        pass

    def release(self):
        """Release camera resources."""
        pass
```

### Display Plugin Interface

```python
class DisplayPlugin:
    def initialize(self):
        """Initialize the display. Return True if successful."""
        pass

    def show_frame(self, bitmap):
        """Show a bitmap frame."""
        pass

    def is_active(self):
        """Check if display is active."""
        pass

    def cleanup(self):
        """Clean up display resources."""
        pass
```

## Benefits

1. **Self-Contained**: Modules manage their own plugin requirements
2. **Scalable**: Easy to add new plugin types (audio, radar, etc.)
3. **Flexible**: Dynamic plugin switching and configuration
4. **Reusable**: Plugins work as black-box APIs
5. **Testable**: Test mode allows development without hardware
6. **Modular**: Clean separation of concerns
7. **Performance**: Direct bitmap output without display overhead
8. **Extensibility**: Easy to add new modules without touching existing code

## Migration from Old System

The old `inference_current.py` can be kept for reference but is no longer needed. The new system:

- Extracts camera logic into reusable plugins
- Converts inference logic into focused modules
- Simplifies the main runner to just orchestration
- Maintains the same ESP32 communication interface

## Testing

Run the test script to verify everything works:

```bash
python src/test_modular_architecture.py
```

This will test:

- Base module interface
- CLIP zero-shot module
- Camera plugins
- Display plugins
- ESP32 client functions

## Advanced Usage

### Dynamic Plugin Switching

```python
module = CLIPZeroShotModule()

# Start with test camera
module.set_camera_type('test', width=640, height=480)
module.set_display_type('console', scale=0.5)

# Initialize plugins
module.initialize_plugins()

# Later, switch to real camera
module.set_camera_type('droidcam', device_id=1)
module.set_display_type('cv2', scale=4)
```

### Adding New Plugin Types (Scalable Approach)

To add a new plugin type (e.g., radar):

1. **Create the plugin file** `src/plugins/radar_plugins.py`:

```python
def create_radar_plugin(radar_type, **kwargs):
    """Factory function to create radar plugins."""
    if radar_type == 'test':
        return TestRadarPlugin(**kwargs)
    else:
        raise ValueError(f"Unknown radar type: {radar_type}")
```

2. **Register in plugin registry** `src/plugins/plugin_registry.py`:

```python
from .radar_plugins import create_radar_plugin

PLUGIN_REGISTRY = {
    'camera': create_camera_plugin,
    'display': create_display_plugin,
    'audio': create_audio_plugin,
    'radar': create_radar_plugin,  # Add this line
}
```

3. **Use in modules** - no changes to BaseModule needed!

```python
def get_plugin_requirements(self):
    return {
        'radar': {
            'type': 'test',
            'config': {'range': 100}
        }
    }
```

**That's it!** The system automatically handles the new plugin type.

## Testing

Run the test scripts to verify the system:

```bash
# Test basic modular architecture
python src/test_modular_architecture.py

# Test plugin scalability
python src/test_plugin_scalability.py
```

## Future Enhancements

- Add more modules (activity classifier, pose detection, etc.)
- Add configuration files for modules
- Add performance monitoring
- Add module hot-swapping
- Add more plugin types (radar, sensors, network, storage, etc.)
