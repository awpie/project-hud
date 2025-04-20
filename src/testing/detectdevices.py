import cv2

def check_camera(device_id):
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        return None
    
    # Get camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Try to read a frame
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return {
            'device_id': device_id,
            'resolution': (width, height),
            'fps': fps,
            'frame_shape': frame.shape if frame is not None else None
        }
    return None

# Check first 10 devices
print("Checking available cameras...")
for i in range(2):
    info = check_camera(i)
    if info:
        print(f"\nDevice {i}:")
        print(f"Resolution: {info['resolution']}")
        print(f"FPS: {info['fps']}")
        print(f"Frame shape: {info['frame_shape']}")
        
        # Show a test frame
        cap = cv2.VideoCapture(i)
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Device {i}", frame)
            cv2.waitKey(1000)  # Show for 1 second
        cap.release()

cv2.destroyAllWindows()