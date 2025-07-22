import cv2
import numpy as np
import urllib.request
import urllib.error
import threading
import time
from urllib.parse import urlparse

class CameraPlugin:
    """Base class for camera plugins."""
    
    def __init__(self):
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the camera. Override in subclasses."""
        pass
    
    def read(self):
        """Read a frame. Override in subclasses."""
        pass
    
    def is_opened(self):
        """Check if camera is opened. Override in subclasses."""
        pass
    
    def release(self):
        """Release camera resources. Override in subclasses."""
        pass

class DroidCamPlugin(CameraPlugin):
    """Plugin for DroidCam camera."""
    
    def __init__(self, device_id=1, width=1280, height=720):
        super().__init__()
        self.cap = None
        self.device_id = device_id
        self.width = width
        self.height = height
    
    def initialize(self):
        """Initialize DroidCam connection."""
        print(f"Attempting to connect to DroidCam on device {self.device_id}...")
        
        # Try the specified device first, then fallback to others
        device_ids = [self.device_id]
        if self.device_id != 1:
            device_ids.append(1)
        if self.device_id != 2:
            device_ids.append(2)
        if self.device_id != 0:
            device_ids.append(0)
        
        for device_id in device_ids:
            self.cap = cv2.VideoCapture(device_id)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                print(f"Successfully connected to DroidCam on device {device_id}")
                self.is_initialized = True
                return True
        
        print("Error: Could not connect to DroidCam. Make sure it's running and connected.")
        return False
    
    def read(self):
        """Read a frame from DroidCam."""
        if not self.is_initialized:
            return False, None
        return self.cap.read()
    
    def is_opened(self):
        """Check if DroidCam is opened."""
        return self.is_initialized and self.cap and self.cap.isOpened()
    
    def release(self):
        """Release DroidCam resources."""
        if self.cap:
            self.cap.release()
        self.is_initialized = False

class MJPEGStreamReader(CameraPlugin):
    """Plugin for MJPEG stream (ESP32-CAM)."""
    
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.stream = None
        self.current_frame = None
        self.running = False
        self.thread = None
    
    def initialize(self):
        """Initialize MJPEG stream connection."""
        try:
            print(f"Opening ESP32-CAM stream with VLC-compatible headers...")
            request = urllib.request.Request(self.url)
            request.add_header('User-Agent', 'VLC/3.0.0')
            request.add_header('Accept', 'multipart/x-mixed-replace,image/jpeg')
            request.add_header('Connection', 'keep-alive')
            
            self.stream = urllib.request.urlopen(request, timeout=15)
            print(f"Stream opened successfully. Content-Type: {self.stream.headers.get('Content-Type', 'Unknown')}")
            
            self.running = True
            self.thread = threading.Thread(target=self._read_stream)
            self.thread.daemon = True
            self.thread.start()
            
            # Wait for first frame
            for i in range(50):  # Wait up to 5 seconds
                time.sleep(0.1)
                ret, frame = self.read()
                if ret and frame is not None:
                    print("Successfully connected to MJPEG stream")
                    self.is_initialized = True
                    return True
            
            print("Error: Could not get frames from MJPEG stream")
            return False
            
        except Exception as e:
            print(f"Error opening MJPEG stream: {e}")
            return False
    
    def _read_stream(self):
        """Read MJPEG stream in a separate thread."""
        buffer = b''
        consecutive_failures = 0
        max_failures = 10
        
        # Get boundary from content-type header
        boundary = None
        content_type = self.stream.headers.get('Content-Type', '')
        if 'boundary=' in content_type:
            boundary = content_type.split('boundary=')[1].encode()
        
        while self.running and consecutive_failures < max_failures:
            try:
                chunk = self.stream.read(4096)
                if not chunk:
                    break
                    
                buffer += chunk
                consecutive_failures = 0
                
                if boundary:
                    # Handle multipart boundary format
                    boundary_marker = b'--' + boundary
                    
                    while boundary_marker in buffer:
                        boundary_start = buffer.find(boundary_marker)
                        if boundary_start == -1:
                            break
                            
                        next_boundary = buffer.find(boundary_marker, boundary_start + len(boundary_marker))
                        if next_boundary == -1:
                            break
                        
                        frame_data = buffer[boundary_start:next_boundary]
                        jpeg_start = frame_data.find(b'\xff\xd8')
                        jpeg_end = frame_data.find(b'\xff\xd9')
                        
                        if jpeg_start != -1 and jpeg_end != -1 and jpeg_end > jpeg_start:
                            jpeg_data = frame_data[jpeg_start:jpeg_end + 2]
                            
                            try:
                                nparr = np.frombuffer(jpeg_data, np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                if frame is not None and frame.size > 0:
                                    self.current_frame = frame
                            except Exception as e:
                                print(f"Error decoding JPEG frame: {e}")
                        
                        buffer = buffer[next_boundary:]
                else:
                    # Fallback: Direct JPEG stream
                    start_marker = b'\xff\xd8'
                    end_marker = b'\xff\xd9'
                    
                    while True:
                        start_idx = buffer.find(start_marker)
                        if start_idx == -1:
                            break
                            
                        end_idx = buffer.find(end_marker, start_idx + 2)
                        if end_idx == -1:
                            break
                            
                        jpeg_data = buffer[start_idx:end_idx + 2]
                        
                        try:
                            nparr = np.frombuffer(jpeg_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None and frame.size > 0:
                                self.current_frame = frame
                        except Exception as e:
                            print(f"Error decoding JPEG frame: {e}")
                        
                        buffer = buffer[end_idx + 2:]
                        
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures < max_failures:
                    time.sleep(1)
        
        self.running = False
    
    def read(self):
        """Read the current frame."""
        if self.current_frame is not None:
            return True, self.current_frame.copy()
        return False, None
    
    def is_opened(self):
        """Check if stream is opened and running."""
        return self.is_initialized and self.running and self.thread and self.thread.is_alive()
    
    def release(self):
        """Release the stream."""
        print("Releasing MJPEG stream...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        if self.stream:
            try:
                self.stream.close()
            except:
                pass
        self.is_initialized = False

class TestCameraPlugin(CameraPlugin):
    """Test camera plugin that generates dummy frames for testing."""
    
    def __init__(self, width=640, height=480):
        super().__init__()
        self.width = width
        self.height = height
        self.frame_count = 0
        self.is_initialized = True
    
    def initialize(self):
        """Initialize test camera."""
        print("✓ Test camera initialized (generating dummy frames)")
        return True
    
    def read(self):
        """Generate a dummy test frame."""
        # Create a test frame with some patterns
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add some animated patterns
        self.frame_count += 1
        
        # Red rectangle that moves
        x = (self.frame_count * 5) % (self.width - 100)
        frame[100:200, x:x+100] = [0, 0, 255]  # Red in BGR
        
        # Green rectangle
        frame[250:350, 400:600] = [0, 255, 0]  # Green in BGR
        
        # Blue rectangle
        frame[350:450, 50:250] = [255, 0, 0]   # Blue in BGR
        
        # Add some text
        cv2.putText(frame, f"Test Frame {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return True, frame
    
    def is_opened(self):
        """Test camera is always open."""
        return True
    
    def release(self):
        """Release test camera."""
        print("Test camera released")
    
    def cleanup(self):
        """Clean up test camera resources."""
        self.is_initialized = False
        print("Test camera cleaned up")

def create_camera_plugin(camera_type, **kwargs):
    """Factory function to create camera plugins."""
    if camera_type == 'droidcam':
        return DroidCamPlugin(**kwargs)
    elif camera_type == 'url':
        url = kwargs.get('url', "http://192.168.0.25:81/stream")
        return MJPEGStreamReader(url)
    elif camera_type == 'test':
        return TestCameraPlugin(**kwargs)
    else:
        raise ValueError(f"Unknown camera type: {camera_type}") 