import requests
import json
import time
import threading
from datetime import datetime
from activity_mapping import activity_to_int

esp32_ip = "192.168.0.233"  # Replace with actual IP

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
    
    # Get current time
    current_time = datetime.now()
    time_str = current_time.strftime("%H:%M")  # 24-hour format HH:MM
    
    # Create JSON packet
    packet = {
        "activity": activity_int,
        "xp_progress": xp_progress,
        "level": current_level,
        "time": time_str,
        "timestamp": current_time.timestamp()  # Unix timestamp for reference
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
            "time": packet["time"],
            "status": activity_name.capitalize()
        }
        
        # Send as JSON POST request to /update endpoint
        r = requests.post(
            f"http://{esp32_ip}/update", 
            json=json_payload,
            timeout=1.0,  # Increased from 0.1 to 1.0 seconds
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
                "time": packet["time"],
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

def shutdown():
    requests.post(f"http://{esp32_ip}/shutdown")

if __name__ == "__main__":
    # Test the packet creation (you'll need to provide an ActivityDisplay instance)
    print("Testing packet creation...")
    
    # Example packet structure
    example_packet = {
        "activity": 0,      # coding
        "xp_progress": 75,  # 75 XP in current level
        "level": 3,         # Level 3
        "time": "14:30",    # 2:30 PM
        "timestamp": time.time()
    }
    
    print("Example packet:")
    print(json.dumps(example_packet, indent=2))
    
    # Test sending simple classification
    for i in range(3):
        send_classification(2)  # Send reading activity
        time.sleep(1)