import cv2
import numpy as np
from PIL import Image
import threading
import time

class DisplayPlugin:
    """Base class for display plugins."""
    
    def __init__(self):
        self.is_initialized = False
        self.window_name = "HUD Preview"
    
    def initialize(self):
        """Initialize the display. Override in subclasses."""
        pass
    
    def show_frame(self, bitmap):
        """Show a bitmap frame. Override in subclasses."""
        pass
    
    def is_active(self):
        """Check if display is active. Override in subclasses."""
        pass
    
    def cleanup(self):
        """Clean up display resources. Override in subclasses."""
        pass

class CV2DisplayPlugin(DisplayPlugin):
    """Plugin for showing bitmap preview using OpenCV window."""
    
    def __init__(self, scale=4, window_name="HUD Preview"):
        super().__init__()
        self.scale = scale  # Scale factor for better visibility
        self.window_name = window_name
        self.window_created = False
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 30  # Target 30 FPS for display
    
    def initialize(self):
        """Initialize CV2 display window."""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 128 * self.scale, 64 * self.scale)
            self.window_created = True
            self.is_initialized = True
            print(f"✓ CV2 preview window created: {self.window_name}")
            return True
        except Exception as e:
            print(f"✗ Failed to create CV2 preview window: {e}")
            return False
    
    def show_frame(self, bitmap):
        """Show bitmap frame in CV2 window."""
        if not self.is_initialized:
            return
        
        # Throttle display updates to maintain performance
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return
        
        try:
            # Convert PIL Image to OpenCV format
            # PIL Image is in 'L' mode (grayscale), convert to BGR for OpenCV
            bitmap_array = np.array(bitmap)
            
            # Convert grayscale to BGR (OpenCV format)
            # Since our bitmap is inverted for OLED (black background, white text),
            # we need to invert it back for normal display
            bitmap_array = 255 - bitmap_array  # Invert back
            bgr_array = cv2.cvtColor(bitmap_array, cv2.COLOR_GRAY2BGR)
            
            # Scale up for better visibility
            if self.scale > 1:
                bgr_array = cv2.resize(bgr_array, (128 * self.scale, 64 * self.scale), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Show the frame
            cv2.imshow(self.window_name, bgr_array)
            cv2.waitKey(1)  # Update window
            
            self.last_frame_time = current_time
            
        except Exception as e:
            print(f"Error showing frame in CV2 window: {e}")
    
    def is_active(self):
        """Check if CV2 window is active."""
        if not self.is_initialized:
            return False
        
        # Check if window still exists
        try:
            # Try to get window property - if window is closed, this will fail
            cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
            return True
        except cv2.error:
            return False
    
    def cleanup(self):
        """Clean up CV2 display resources."""
        if self.window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
        self.is_initialized = False

class ConsoleDisplayPlugin(DisplayPlugin):
    """Plugin for showing bitmap preview in console using ASCII art."""
    
    def __init__(self, scale=0.5):
        super().__init__()
        self.scale = scale  # Scale down for console display
        self.is_initialized = True
        self.last_frame_time = 0
        self.frame_interval = 0.5  # Show frame every 0.5 seconds for better responsiveness
    
    def initialize(self):
        """Initialize console display."""
        print("✓ Console preview enabled (showing every second)")
        return True
    
    def show_frame(self, bitmap):
        """Show bitmap frame in console as ASCII art."""
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return
        
        try:
            # Convert PIL Image to numpy array
            bitmap_array = np.array(bitmap)
            
            # Scale down for console display
            if self.scale < 1.0:
                height, width = bitmap_array.shape
                new_height = int(height * self.scale)
                new_width = int(width * self.scale)
                bitmap_array = cv2.resize(bitmap_array, (new_width, new_height), 
                                        interpolation=cv2.INTER_NEAREST)
            
            # Convert to ASCII art
            ascii_chars = " .:-=+*#%@"
            ascii_frame = []
            
            for row in bitmap_array:
                ascii_row = ""
                for pixel in row:
                    # Convert pixel value (0-255) to ASCII character
                    char_index = int((255 - pixel) / 255 * (len(ascii_chars) - 1))
                    ascii_row += ascii_chars[char_index]
                ascii_frame.append(ascii_row)
            
            # Clear console and show frame
            print("\033[2J\033[H")  # Clear screen and move cursor to top
            print("HUD Preview (Console):")
            print("=" * 50)
            for row in ascii_frame:
                print(row)
            print("=" * 50)
            
            self.last_frame_time = current_time
            
        except Exception as e:
            print(f"Error showing frame in console: {e}")
    
    def is_active(self):
        """Console display is always active."""
        return True
    
    def cleanup(self):
        """Clean up console display."""
        print("\nConsole preview stopped.")

class NoDisplayPlugin(DisplayPlugin):
    """Plugin for no display (headless mode)."""
    
    def __init__(self):
        super().__init__()
        self.is_initialized = True
    
    def initialize(self):
        """Initialize no display mode."""
        print("✓ Running in headless mode (no preview)")
        return True
    
    def show_frame(self, bitmap):
        """Do nothing - no display."""
        pass
    
    def is_active(self):
        """Always active in headless mode."""
        return True
    
    def cleanup(self):
        """No cleanup needed."""
        pass

def create_display_plugin(display_type, **kwargs):
    """Factory function to create display plugins."""
    if display_type == 'cv2':
        return CV2DisplayPlugin(**kwargs)
    elif display_type == 'console':
        return ConsoleDisplayPlugin(**kwargs)
    elif display_type == 'none':
        return NoDisplayPlugin()
    else:
        raise ValueError(f"Unknown display type: {display_type}") 