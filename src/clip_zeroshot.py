# Zero-shot CLIP classification using direct image-to-text comparison
import cv2
import torch
from PIL import Image
import numpy as np
from collections import deque
import open_clip
import logging
from activity_display import ActivityDisplay
import urllib.request
import urllib.error
import threading
import time
from urllib.parse import urlparse

#cam stream
cam_stream = "http://192.168.0.25:81/stream"

# Define text prompts for each class


CLASS_PROMPT = {
    'coding': [
        'point-of-view of coding on a computer',
        'person typing on a keyboard',
        'computer screen with code',
        'programming on a laptop'
    ],
    'eating': [
        'point-of-view of eating food',
        'person eating a meal',
        'dining at a table',
        'consuming food'
    ],
    'reading': [
        'point-of-view of reading a book',
        'person reading a book',
        'holding and reading a book',
        'looking at a book'
    ],
    'piano': [
        'playing the piano',
        'person at a piano',
        'piano keyboard',
        'musician playing piano'
    ],
    'nature': [
        'nature',
        'outdoor scene',
        'natural landscape',
        'outdoors'
    ],
    'idle': [
        'view of random objects',
        'view of unmoving things',
        'furniture',
        'walls',
        'nothing interesting', 
        'view of a kitchen',
        'view of a bedroom',
        'view of a bathroom',
        'view of a living room',
        'view of a dining room',
        'view of a hallway',
        'view of a general scene'
    ]
}

# Function to get smoothed prediction
def get_smoothed_prediction(prediction_buffer):
    if not prediction_buffer:
        return None
    
    # Convert buffer to numpy array for easier manipulation
    buffer_array = np.array(prediction_buffer)
    
    # Apply exponential weighting (more recent predictions have higher weight)
    weights = np.exp(np.linspace(-1, 0, len(buffer_array)))
    weights = weights / weights.sum()
    
    # Calculate weighted average
    weighted_avg = np.average(buffer_array, weights=weights, axis=0)
    
    return weighted_avg

# Function to get user's camera source choice
def get_camera_choice():
    """Get user's choice for camera input source."""
    print("\nSelect camera input source:")
    print("1. DroidCam (USB/WiFi connected Android device)")
    print("2. URL Camera Stream (http://192.168.0.25:81/stream)")
    print("3. Test URL Connection (debug)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()
            if choice == '1':
                return 'droidcam'
            elif choice == '2':
                return 'url'
            elif choice == '3':
                return 'test_url'
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nProgram interrupted by user.")
            return None
        except Exception as e:
            print(f"Error reading input: {e}")
            return None

class MJPEGStreamReader:
    """Custom MJPEG stream reader using urllib for better compatibility."""
    
    def __init__(self, url):
        self.url = url
        self.stream = None
        self.current_frame = None
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the MJPEG stream reading thread."""
        try:
            # Use the working VLC user agent approach from the diagnostic
            print(f"Opening ESP32-CAM stream with VLC-compatible headers...")
            request = urllib.request.Request(self.url)
            request.add_header('User-Agent', 'VLC/3.0.0')
            request.add_header('Accept', 'multipart/x-mixed-replace,image/jpeg')
            request.add_header('Connection', 'keep-alive')
            
            self.stream = urllib.request.urlopen(request, timeout=15)
            print(f"Stream opened successfully. Content-Type: {self.stream.headers.get('Content-Type', 'Unknown')}")
            
            # Check if we got the expected boundary format
            content_type = self.stream.headers.get('Content-Type', '')
            if 'boundary=' in content_type:
                boundary = content_type.split('boundary=')[1]
                print(f"Detected boundary: {boundary}")
            
            self.running = True
            self.thread = threading.Thread(target=self._read_stream)
            self.thread.daemon = True
            self.thread.start()
            return True
            
        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code}: {e.reason}")
            return False
        except urllib.error.URLError as e:
            print(f"URL Error: {e.reason}")
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
            print(f"Using boundary: {boundary.decode()}")
        
        print("Starting ESP32-CAM stream reading thread...")
        
        while self.running and consecutive_failures < max_failures:
            try:
                chunk = self.stream.read(4096)  # Increased chunk size
                if not chunk:
                    print("MJPEG stream ended (no more data)")
                    break
                    
                buffer += chunk
                consecutive_failures = 0  # Reset failure counter on successful read
                
                # Handle multipart boundary format or direct JPEG
                if boundary:
                    # ESP32-CAM uses multipart format with boundaries
                    boundary_marker = b'--' + boundary
                    
                    while boundary_marker in buffer:
                        # Find the boundary
                        boundary_start = buffer.find(boundary_marker)
                        if boundary_start == -1:
                            break
                            
                        # Look for the next boundary to get the complete frame
                        next_boundary = buffer.find(boundary_marker, boundary_start + len(boundary_marker))
                        if next_boundary == -1:
                            # Incomplete frame, wait for more data
                            break
                        
                        # Extract the frame data between boundaries
                        frame_data = buffer[boundary_start:next_boundary]
                        
                        # Look for JPEG data within this frame
                        jpeg_start = frame_data.find(b'\xff\xd8')
                        jpeg_end = frame_data.find(b'\xff\xd9')
                        
                        if jpeg_start != -1 and jpeg_end != -1 and jpeg_end > jpeg_start:
                            jpeg_data = frame_data[jpeg_start:jpeg_end + 2]
                            
                            # Convert JPEG bytes to OpenCV frame
                            try:
                                nparr = np.frombuffer(jpeg_data, np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                if frame is not None and frame.size > 0:
                                    self.current_frame = frame
                                    # print(f"Decoded ESP32-CAM frame: {frame.shape}")  # Debug info
                            except Exception as e:
                                print(f"Error decoding JPEG frame: {e}")
                        
                        # Remove processed data from buffer
                        buffer = buffer[next_boundary:]
                else:
                    # Fallback: Direct JPEG stream (no boundaries)
                    start_marker = b'\xff\xd8'  # JPEG start
                    end_marker = b'\xff\xd9'    # JPEG end
                    
                    while True:
                        start_idx = buffer.find(start_marker)
                        if start_idx == -1:
                            break
                            
                        # Found start of JPEG, look for end
                        end_idx = buffer.find(end_marker, start_idx + 2)
                        if end_idx == -1:
                            # Incomplete JPEG, wait for more data
                            break
                            
                        # Found complete JPEG
                        jpeg_data = buffer[start_idx:end_idx + 2]
                        
                        # Convert JPEG bytes to OpenCV frame
                        try:
                            nparr = np.frombuffer(jpeg_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None and frame.size > 0:
                                self.current_frame = frame
                        except Exception as e:
                            print(f"Error decoding JPEG frame: {e}")
                        
                        # Remove processed data from buffer
                        buffer = buffer[end_idx + 2:]
                        
            except Exception as e:
                consecutive_failures += 1
                print(f"Error reading MJPEG stream (attempt {consecutive_failures}): {e}")
                if consecutive_failures < max_failures:
                    time.sleep(1)  # Wait before retrying
                
        print(f"ESP32-CAM stream reading thread ended. Failures: {consecutive_failures}")
        self.running = False
    
    def read(self):
        """Read the current frame."""
        if self.current_frame is not None:
            return True, self.current_frame.copy()
        return False, None
    
    def isOpened(self):
        """Check if stream is opened and running."""
        return self.running and self.thread is not None and self.thread.is_alive()
    
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

def test_url_connection():
    """Test the URL connection with various methods."""
    print(f"\n=== Testing URL Connection: {cam_stream} ===")
    
    # Test 1: Basic connectivity
    try:
        import socket
        from urllib.parse import urlparse
        
        parsed = urlparse(cam_stream)
        host = parsed.hostname
        port = parsed.port or 80
        
        print(f"1. Testing basic connectivity to {host}:{port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print("   ✓ Host is reachable")
        else:
            print(f"   ✗ Host unreachable (error code: {result})")
            return
            
    except Exception as e:
        print(f"   ✗ Basic connectivity test failed: {e}")
        return
    
    # Test 2: ESP32-CAM specific headers (same as our stream reader)
    print("2. Testing with ESP32-CAM browser headers...")
    try:
        parsed = urlparse(cam_stream)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # First, access the main page like a browser would
        print("   Step 1: Accessing main ESP32-CAM page...")
        main_request = urllib.request.Request(base_url + "/")
        main_request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36')
        main_request.add_header('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7')
        main_request.add_header('Accept-Language', 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7')
        main_request.add_header('Connection', 'keep-alive')
        main_request.add_header('Host', parsed.netloc)
        
        main_response = urllib.request.urlopen(main_request, timeout=5)
        print(f"   ✓ Main page accessed successfully (HTTP {main_response.getcode()})")
        main_response.close()
        
        # Small delay to mimic human interaction
        time.sleep(0.5)
        
        # Now request the stream with proper headers
        print("   Step 2: Requesting stream with browser headers...")
        request = urllib.request.Request(cam_stream)
        request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36')
        request.add_header('Accept', 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8')
        request.add_header('Accept-Encoding', 'gzip, deflate')
        request.add_header('Accept-Language', 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7')
        request.add_header('Connection', 'keep-alive')
        request.add_header('Origin', base_url)
        request.add_header('Referer', f"{base_url}/")
        request.add_header('Sec-GPC', '1')
        request.add_header('Host', parsed.netloc)
        
        print("   Sending stream request...")
        response = urllib.request.urlopen(request, timeout=10)
        print(f"   ✓ ESP32-CAM stream request successful!")
        print(f"   Response code: {response.getcode()}")
        print(f"   Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
        
        # Check for expected boundary
        content_type = response.headers.get('Content-Type', '')
        if 'boundary=' in content_type:
            boundary = content_type.split('boundary=')[1]
            print(f"   ✓ Detected boundary: {boundary}")
        
        # Try to read some data
        data = response.read(4096)
        print(f"   Data received: {len(data)} bytes")
        
        # Look for JPEG markers
        if b'\xff\xd8' in data:
            print("   ✓ JPEG data detected in stream")
        else:
            print("   ⚠ No JPEG markers found in initial data")
            print(f"   First 100 bytes: {data[:100]}")
        
        response.close()
        print("   ✓ ESP32-CAM connection test successful!")
        
    except urllib.error.HTTPError as e:
        print(f"   ✗ HTTP Error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        print(f"   ✗ URL Error: {e.reason}")
    except Exception as e:
        print(f"   ✗ ESP32-CAM headers request failed: {e}")
        return
    
    # Test 3: Raw HTTP request via socket
    print("3. Testing raw HTTP request via socket...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((host, port))
        
        # Send minimal HTTP request
        request = f"GET {parsed.path} HTTP/1.1\r\nHost: {host}\r\n\r\n"
        sock.send(request.encode())
        
        # Try to receive response
        response = sock.recv(1024)
        if response:
            print("   ✓ Raw HTTP request successful")
            response_str = response.decode('utf-8', errors='ignore')
            print(f"   Response: {response_str[:200]}...")
        else:
            print("   ✗ No response received")
        
        sock.close()
        
    except Exception as e:
        print(f"   ✗ Raw HTTP request failed: {e}")
    
    # Test 4: HTTP request with minimal headers
    print("4. Testing basic HTTP request with minimal headers...")
    try:
        request = urllib.request.Request(cam_stream)
        response = urllib.request.urlopen(request, timeout=5)
        print(f"   ✓ Basic HTTP request successful")
        print(f"   Response code: {response.getcode()}")
        print(f"   Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
        
        # Read a small amount of data
        data = response.read(1024)
        print(f"   First 1KB received: {len(data)} bytes")
        response.close()
        
    except urllib.error.HTTPError as e:
        print(f"   ✗ HTTP Error {e.code}: {e.reason}")
        if e.code == 401:
            print("   (This requires authentication)")
        elif e.code == 403:
            print("   (Access forbidden)")
        elif e.code == 404:
            print("   (URL path not found)")
    except urllib.error.URLError as e:
        print(f"   ✗ URL Error: {e.reason}")
    except Exception as e:
        print(f"   ✗ Request failed: {e}")
    
    # Test 5: Different user agents
    print("5. Testing with different User-Agent strings...")
    user_agents = [
        "VLC/3.0.0",
        "curl/7.68.0",
        "Mozilla/5.0 (compatible; MJPEG-Client)",
        "Python-urllib/3.8"
    ]
    
    for ua in user_agents:
        try:
            request = urllib.request.Request(cam_stream)
            request.add_header('User-Agent', ua)
            response = urllib.request.urlopen(request, timeout=5)
            print(f"   ✓ Success with User-Agent: {ua}")
            response.close()
            break
        except Exception as e:
            print(f"   ✗ Failed with User-Agent '{ua}': {type(e).__name__}")
    
    print("\n=== URL Connection Test Complete ===")
    print("✓ If Test 2 (ESP32-CAM browser headers) passed, try option 2 to start streaming!")
    print("💡 Tips:")
    print("- Keep the ESP32-CAM web interface open in your browser")
    print("- Make sure the stream is active in the web UI before starting Python")
    print("- The ESP32-CAM might need the web interface to be 'primed' first")

def initialize_camera(camera_type):
    """Initialize camera based on the selected type."""
    cap = None
    
    if camera_type == 'test_url':
        test_url_connection()
        return None
    elif camera_type == 'droidcam':
        print("Attempting to connect to DroidCam...")
        # Try to open DroidCam
        for device_id in [1, 2, 0]:  # Added device 0 as fallback
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increased resolution
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print(f"Successfully connected to DroidCam on device {device_id}")
                return cap
        print("Error: Could not connect to DroidCam. Make sure it's running and connected.")
        return None
        
    elif camera_type == 'url':
        print(f"Attempting to connect to MJPEG stream: {cam_stream}")
        cap = MJPEGStreamReader(cam_stream)
        if cap.start():
            # Wait a moment for the first frame
            for i in range(50):  # Wait up to 5 seconds
                time.sleep(0.1)
                ret, frame = cap.read()
                if ret and frame is not None:
                    print("Successfully connected to MJPEG stream")
                    return cap
            print("Error: Could not get frames from MJPEG stream")
            cap.release()
            return None
        print(f"Error: Could not connect to MJPEG stream at {cam_stream}")
        return None
    
    return None

def run_zeroshot_inference():
    # Get user's camera choice
    camera_type = get_camera_choice()
    if camera_type is None:
        print("No camera source selected. Exiting.")
        return
    
    # Initialize activity display
    display = ActivityDisplay()
    
    # Initialize CLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device)
    model.eval()

    # Encode text prompts
    text_features = {}
    for class_name, prompts in CLASS_PROMPT.items():
        # Encode all prompts for this class
        prompt_features = []
        for prompt in prompts:
            text_tokens = tokenizer([prompt]).to(device)
            with torch.no_grad():
                text_feature = model.encode_text(text_tokens)
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                prompt_features.append(text_feature)
        text_features[class_name] = prompt_features

    # Temporal smoothing buffer
    buffer_size = 30  # Increased from 15 to 30 for more stable predictions
    prediction_buffer = deque(maxlen=buffer_size)

    try:
        # Initialize camera based on user choice
        cap = initialize_camera(camera_type)
        if cap is None:
            print("Failed to initialize camera. Exiting.")
            return

        camera_source = "DroidCam" if camera_type == 'droidcam' else f"URL Stream ({cam_stream})"
        print(f"{camera_source} started. Press 'q' to quit.")

        # Create a resizable window for camera view
        cv2.namedWindow("CLIP Zero-shot Classification", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CLIP Zero-shot Classification", 1280, 720)

        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    print(f"Error: Could not read frame from {camera_source}.")
                    break

                # Check if frame is valid
                if frame is None or frame.size == 0:
                    continue

                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Get CLIP features
                with torch.no_grad():
                    image_features = model.encode_image(preprocess(pil_image).unsqueeze(0).to(device))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarities with all text prompts
                    similarities = {}
                    for class_name, prompt_features in text_features.items():
                        # Calculate similarity with each prompt and average
                        class_similarities = []
                        for text_feature in prompt_features:
                            similarity = torch.nn.functional.cosine_similarity(image_features, text_feature)
                            class_similarities.append(similarity.item())
                        similarities[class_name] = np.mean(class_similarities)
                    
                    # Convert to array and add to buffer
                    similarity_array = np.array(list(similarities.values()))
                    prediction_buffer.append(similarity_array)

                # Get smoothed prediction
                smoothed_similarities = get_smoothed_prediction(prediction_buffer)
                if smoothed_similarities is not None:
                    predicted_idx = np.argmax(smoothed_similarities)
                    predicted_label = list(CLASS_PROMPT.keys())[predicted_idx]
                    confidence = smoothed_similarities[predicted_idx]
                    
                    # Update activity display
                    display.update(predicted_label, confidence)

                    # Create a copy of the frame for display
                    display_frame = frame.copy()
                    
                    # Add a semi-transparent overlay for text
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (300, display_frame.shape[0]), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

                    # Display predictions in two columns
                    y_offset = 30
                    for i, (class_name, similarity) in enumerate(zip(CLASS_PROMPT.keys(), smoothed_similarities)):
                        similarity_text = f"{class_name}: {similarity:.3f}"
                        color = (0, 255, 0) if i == predicted_idx else (255, 255, 255)
                        cv2.putText(display_frame, similarity_text, (10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        y_offset += 20

                    cv2.imshow("CLIP Zero-shot Classification", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except KeyboardInterrupt:
                print("\nInference interrupted by user. Exiting gracefully...")
                break
            except Exception as e:
                print(f"\nAn error occurred during inference: {e}")
                continue

    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Cleanup
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        display.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Resources cleaned up.")

if __name__ == "__main__":
    run_zeroshot_inference() 