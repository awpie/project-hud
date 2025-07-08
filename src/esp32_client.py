"""
Client for ESP32 server.

This file contains the client code for the ESP32 server.
It is used to send activity data to the ESP32 server.
Currently supports JSON packet and bitmap image.
"""

import requests
import json
import time
import threading
from datetime import datetime
from activity_mapping import activity_to_int
from PIL import Image, ImageDraw, ImageFont
import io
import base64

esp32_ip = "192.168.0.233"  # Replace with actual IP

# OLED display configuration
OLED_WIDTH = 128
OLED_HEIGHT = 64

def create_hud_image(activity_display):
    """Create a complete HUD image for OLED display.
    
    Args:
        activity_display: ActivityDisplay instance with current activity data
        
    Returns:
        PIL.Image: 128x64 monochrome image ready for OLED (after bitmap conversion)
    """
    # Create a new image with white background (will be inverted for OLED)
    image = Image.new('L', (OLED_WIDTH, OLED_HEIGHT), 255)
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        # Try to use a small font that fits OLED display
        font_small = ImageFont.truetype("arial.ttf", 8)
        font_medium = ImageFont.truetype("arial.ttf", 10)
        font_large = ImageFont.truetype("arial.ttf", 12)
    except:
        # Fallback to default font
        font_small = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_large = ImageFont.load_default()
    
    # Get activity data
    current_activity = activity_display.get_current_activity() or "No Activity"
    confidence = activity_display.get_current_confidence()
    duration = activity_display.get_current_duration()
    activity_tracker = activity_display.get_activity_tracker()
    current_xp = activity_tracker.get_current_xp()
    current_level = current_xp // 100
    xp_progress = current_xp % 100
    
    # Draw activity name (top left)
    activity_text = f"{current_activity[:10]}"  # Truncate if too long
    draw.text((2, 2), activity_text, fill=0, font=font_medium)
    
    # Draw confidence (top right)
    confidence_text = f"{confidence:.1%}"
    confidence_bbox = draw.textbbox((0, 0), confidence_text, font=font_small)
    confidence_width = confidence_bbox[2] - confidence_bbox[0]
    draw.text((OLED_WIDTH - confidence_width - 2, 2), confidence_text, fill=0, font=font_small)
    
    # Draw timer (center top)
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    timer_text = f"{minutes:02d}:{seconds:02d}"
    timer_bbox = draw.textbbox((0, 0), timer_text, font=font_large)
    timer_width = timer_bbox[2] - timer_bbox[0]
    timer_x = (OLED_WIDTH - timer_width) // 2
    draw.text((timer_x, 15), timer_text, fill=0, font=font_large)
    
    # Draw XP bar (middle)
    bar_y = 35
    bar_height = 8
    bar_width = OLED_WIDTH - 4
    
    # XP bar background
    draw.rectangle([(2, bar_y), (OLED_WIDTH - 2, bar_y + bar_height)], outline=0, width=1)
    
    # XP progress fill
    progress_width = int((xp_progress / 100) * (bar_width - 2))
    if progress_width > 0:
        draw.rectangle([(3, bar_y + 1), (3 + progress_width, bar_y + bar_height - 1)], fill=0)
    
    # Draw XP and level info (bottom)
    xp_text = f"XP: {current_xp}"
    level_text = f"Lv.{current_level}"
    
    draw.text((2, bar_y + bar_height + 5), xp_text, fill=0, font=font_small)
    
    level_bbox = draw.textbbox((0, 0), level_text, font=font_small)
    level_width = level_bbox[2] - level_bbox[0]
    draw.text((OLED_WIDTH - level_width - 2, bar_y + bar_height + 5), level_text, fill=0, font=font_small)
    
    # Draw pending activity if there is one
    if activity_tracker.pending_activity is not None:
        pending_time = time.time() - activity_tracker.pending_start_time
        pending_text = f"-> {activity_tracker.pending_activity[:8]} ({pending_time:.1f}s)"
        draw.text((2, OLED_HEIGHT - 12), pending_text, fill=0, font=font_small)
    
    # Invert the image for OLED (black background, white text)
    image = Image.eval(image, lambda x: 255 - x)
    
    return image

def image_to_bitmap(image):
    """Convert PIL image to bitmap format for ESP32.
    
    Args:
        image: PIL Image object (128x64 monochrome)
        
    Returns:
        bytes: Bitmap data in format suitable for ESP32 OLED
    """
    # Ensure image is 128x64 and monochrome
    if image.size != (OLED_WIDTH, OLED_HEIGHT):
        image = image.resize((OLED_WIDTH, OLED_HEIGHT))
    
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to bitmap format
    # Each byte represents 8 vertical pixels
    bitmap_data = bytearray()
    
    for x in range(OLED_WIDTH):
        for page in range(OLED_HEIGHT // 8):
            byte = 0
            for bit in range(8):
                y = page * 8 + bit
                if y < OLED_HEIGHT:
                    # Get pixel value (0 or 255) and convert to bit
                    pixel = image.getpixel((x, y))
                    if pixel > 128:  # White pixel
                        byte |= (1 << bit)
            bitmap_data.append(byte)
    
    return bytes(bitmap_data)

def image_to_base64(image):
    """Convert PIL image to base64 string for HTTP transmission.
    
    Args:
        image: PIL Image object
        
    Returns:
        str: Base64 encoded image data
    """
    # Convert to bitmap first
    bitmap_data = image_to_bitmap(image)
    
    # Encode to base64
    base64_data = base64.b64encode(bitmap_data).decode('utf-8')
    
    return base64_data

def send_bitmap_image(activity_display):
    """Send bitmap image to ESP32.
    
    Args:
        activity_display: ActivityDisplay instance with current activity data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create HUD image
        image = create_hud_image(activity_display)
        
        # Convert to base64
        base64_data = image_to_base64(image)
        
        # Create payload
        payload = {
            "bitmap": base64_data,
            "width": OLED_WIDTH,
            "height": OLED_HEIGHT,
            "timestamp": int(time.time())
        }
        
        # Send to ESP32
        r = requests.post(
            f"http://{esp32_ip}/bitmap", 
            json=payload,
            timeout=2.0,
            headers={'Content-Type': 'application/json'}
        )
        
        if r.status_code == 200:
            print(f"✓ Bitmap image sent successfully")
            return True
        else:
            print(f"✗ ESP32 returned status code: {r.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ ESP32 not responding: {e}")
        return False
    except Exception as e:
        print(f"✗ Error sending bitmap image: {e}")
        return False

def send_bitmap_image_raw(activity_display):
    """Send bitmap image to ESP32 as raw binary data.
    
    Args:
        activity_display: ActivityDisplay instance with current activity data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create HUD image
        image = create_hud_image(activity_display)
        
        # Convert to bitmap
        bitmap_data = image_to_bitmap(image)
        
        # Send raw binary data
        r = requests.post(
            f"http://{esp32_ip}/bitmap_raw", 
            data=bitmap_data,
            timeout=2.0,
            headers={'Content-Type': 'application/octet-stream'}
        )
        
        if r.status_code == 200:
            print(f"✓ Raw bitmap sent successfully ({len(bitmap_data)} bytes)")
            return True
        else:
            print(f"✗ ESP32 returned status code: {r.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ ESP32 not responding: {e}")
        return False
    except Exception as e:
        print(f"✗ Error sending raw bitmap: {e}")
        return False

def save_hud_image(activity_display, filename=None):
    """Save HUD image to file for debugging.
    
    Args:
        activity_display: ActivityDisplay instance with current activity data
        filename: Optional filename, defaults to timestamp-based name
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hud_image_{timestamp}.png"
    
    image = create_hud_image(activity_display)
    image.save(filename)
    print(f"✓ HUD image saved as {filename}")

def print_bitmap_info(activity_display):
    """Print bitmap information for debugging.
    
    Args:
        activity_display: ActivityDisplay instance with current activity data
    """
    image = create_hud_image(activity_display)
    bitmap_data = image_to_bitmap(image)
    
    print("Bitmap Information:")
    print(f"  Image size: {image.size}")
    print(f"  Bitmap size: {len(bitmap_data)} bytes")
    print(f"  Expected size: {OLED_WIDTH * (OLED_HEIGHT // 8)} bytes")
    print(f"  Base64 size: {len(image_to_base64(image))} characters")

def send_classification(classification): 
    try:
        r = requests.get(f"http://{esp32_ip}/update", params={"val": classification}, timeout=1.0)
        if r.status_code == 200:
            print("Sent!")
    except requests.exceptions.RequestException as e:
        print("ESP32 not responding.")

def create_activity_packet(activity_display):
    """Create a JSON packet with activity data for ESP32.
    
    Args:
        activity_display: ActivityDisplay instance with current activity data
        
    Returns:
        dict: JSON packet with activity, XP, level, and time
    """
    # Get current activity and convert to int
    current_activity = activity_display.get_current_activity()
    activity_int = activity_to_int(current_activity) if current_activity else 5  # Default to idle
    
    # Get XP and level data
    activity_tracker = activity_display.get_activity_tracker()
    current_xp = activity_tracker.get_current_xp()
    current_level = current_xp // 100  # Level is XP divided by 100
    xp_progress = current_xp % 100     # XP progress within current level (0-99)
    
    # Create JSON packet
    packet = {
        "activity": activity_int,
        "xp_progress": xp_progress,
        "level": current_level,
    }
    
    return packet

def send_activity_packet(activity_display):
    """Send activity packet to ESP32.
    
    Args:
        activity_display: ActivityDisplay instance with current activity data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        packet = create_activity_packet(activity_display)
        
        # Convert activity int to string for ESP32
        activity_names = {0: "coding", 1: "eating", 2: "reading", 3: "piano", 4: "nature", 5: "idle"}
        activity_name = activity_names.get(packet["activity"], "idle")
        
        # Create JSON payload matching ESP32 server expectations
        json_payload = {
            "class": activity_name,
            "xp": packet["xp_progress"],
            "level": packet["level"],
            "status": activity_name.capitalize()
        }
        
        # Send as JSON POST request to /update endpoint
        r = requests.post(
            f"http://{esp32_ip}/update", 
            json=json_payload,
            timeout=1.0,  # Increased from 0.1 to 1.0 seconds since it was crashing
            headers={'Content-Type': 'application/json'}
        )
        
        if r.status_code == 200:
            print(f"✓ Activity packet sent: {json_payload}")
            return True
        else:
            print(f"✗ ESP32 returned status code: {r.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ ESP32 not responding: {e}")
        return False
    except Exception as e:
        print(f"✗ Error sending activity packet: {e}")
        return False

def send_activity_packet_json(activity_display):
    """Send activity packet to ESP32 as JSON (alternative method).
    
    Args:
        activity_display: ActivityDisplay instance with current activity data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        packet = create_activity_packet(activity_display)
        
        # Send as JSON in request body
        r = requests.post(
            f"http://{esp32_ip}/activity", 
            json=packet,
            timeout=0.1,
            headers={'Content-Type': 'application/json'}
        )
        
        if r.status_code == 200:
            print(f"✓ Activity packet sent (JSON): {packet}")
            return True
        else:
            print(f"✗ ESP32 returned status code: {r.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ ESP32 not responding: {e}")
        return False
    except Exception as e:
        print(f"✗ Error sending activity packet: {e}")
        return False

def print_activity_packet(activity_display):
    """Print the activity packet for debugging.
    
    Args:
        activity_display: ActivityDisplay instance with current activity data
    """
    packet = create_activity_packet(activity_display)
    print("Activity Packet:")
    print(json.dumps(packet, indent=2))

def send_activity_packet_async(activity_display):
    """Send activity packet to ESP32 asynchronously (non-blocking).
    
    Args:
        activity_display: ActivityDisplay instance with current activity data
    """
    def _send_async():
        try:
            packet = create_activity_packet(activity_display)
            
            # Convert activity int to string for ESP32
            activity_names = {0: "coding", 1: "eating", 2: "reading", 3: "piano", 4: "nature", 5: "idle"}
            activity_name = activity_names.get(packet["activity"], "idle")
            
            # Create JSON payload matching ESP32 server expectations
            json_payload = {
                "class": activity_name,
                "xp": packet["xp_progress"],
                "level": packet["level"],
                "status": activity_name.capitalize()
            }
            
            # Send as JSON POST request to /update endpoint
            r = requests.post(
                f"http://{esp32_ip}/update", 
                json=json_payload,
                timeout=2.0,  # Longer timeout for async requests
                headers={'Content-Type': 'application/json'}
            )
            
            if r.status_code == 200:
                print(f"✓ Activity packet sent (async): {json_payload}")
            else:
                print(f"✗ ESP32 returned status code: {r.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"✗ ESP32 not responding (async): {e}")
        except Exception as e:
            print(f"✗ Error sending activity packet (async): {e}")
    
    # Start async thread
    thread = threading.Thread(target=_send_async, daemon=True)
    thread.start()

def send_bitmap_image_async(activity_display):
    """Send bitmap image to ESP32 asynchronously (non-blocking).
    
    Args:
        activity_display: ActivityDisplay instance with current activity data
    """
    def _send_async():
        try:
            # Create HUD image
            image = create_hud_image(activity_display)
            
            # Convert to base64
            base64_data = image_to_base64(image)
            
            # Create payload
            payload = {
                "bitmap": base64_data,
                "width": OLED_WIDTH,
                "height": OLED_HEIGHT,
                "timestamp": int(time.time())
            }
            
            # Send to ESP32
            r = requests.post(
                f"http://{esp32_ip}/bitmap", 
                json=payload,
                timeout=3.0,  # Longer timeout for async requests
                headers={'Content-Type': 'application/json'}
            )
            
            if r.status_code == 200:
                print(f"✓ Bitmap image sent (async)")
            else:
                print(f"✗ ESP32 returned status code: {r.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"✗ ESP32 not responding (async): {e}")
        except Exception as e:
            print(f"✗ Error sending bitmap image (async): {e}")
    
    # Start async thread
    thread = threading.Thread(target=_send_async, daemon=True)
    thread.start()

def shutdown():
    requests.post(f"http://{esp32_ip}/shutdown")

if __name__ == "__main__":
    # Test the packet creation and bitmap functionality
    print("Testing packet creation and bitmap generation...")
    
    # Create a mock ActivityDisplay for testing
    from activity_display import ActivityDisplay
    
    # Create test display
    test_display = ActivityDisplay()
    
    # Simulate some activities
    test_activities = [
        ("coding", 0.85),
        ("eating", 0.72),
        ("reading", 0.91),
        ("piano", 0.68),
        ("nature", 0.45),
        ("idle", 0.23)
    ]
    
    print("\n=== Testing Activity Packet Creation ===")
    
    for i, (activity, confidence) in enumerate(test_activities):
        # Update the display with activity
        test_display.update(activity, confidence)
        
        # Wait a bit to simulate activity duration
        time.sleep(2)
        
        print(f"\n--- Activity: {activity} (confidence: {confidence:.2f}) ---")
        
        # Create and print the packet
        print_activity_packet(test_display)
        
        # Print bitmap info
        print_bitmap_info(test_display)
        
        # Save HUD image for debugging
        save_hud_image(test_display, f"test_hud_{i}.png")
        
        # Try to send to ESP32 (will fail if ESP32 not running)
        print(f"Sending to ESP32...")
        
        # Try both JSON packet and bitmap
        success_json = send_activity_packet(test_display)
        success_bitmap = send_bitmap_image(test_display)
        
        if success_json:
            print("✓ JSON packet sent successfully!")
        else:
            print("✗ Failed to send JSON packet (ESP32 may not be running)")
            
        if success_bitmap:
            print("✓ Bitmap image sent successfully!")
        else:
            print("✗ Failed to send bitmap image (ESP32 may not be running)")
        
        print("-" * 50)
    
    # Test async sending
    print("\n=== Testing Async Bitmap Sending ===")
    test_display.update("coding", 0.95)
    send_bitmap_image_async(test_display)
    print("Async bitmap send initiated...")
    time.sleep(1)
    
    print("\nTesting completed!")