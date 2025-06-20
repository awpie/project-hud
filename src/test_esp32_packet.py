# Test script for ESP32 activity packet functionality
from activity_display import ActivityDisplay
from esp32_client import create_activity_packet, send_activity_packet, print_activity_packet

def test_activity_packet():
    """Test creating and sending activity packets."""
    
    # Create a mock activity display for testing
    display = ActivityDisplay()
    
    # Simulate some activities
    test_activities = [
        ("coding", 0.85),
        ("eating", 0.72),
        ("reading", 0.91),
        ("piano", 0.68),
        ("nature", 0.45),
        ("idle", 0.23)
    ]
    
    print("=== Testing Activity Packet Creation ===")
    
    for activity, confidence in test_activities:
        # Update the display with activity
        display.update(activity, confidence)
        
        # Wait a bit to simulate activity duration
        import time
        time.sleep(2)
        
        # Create and print the packet
        print(f"\n--- Activity: {activity} (confidence: {confidence:.2f}) ---")
        print_activity_packet(display)
        
        # Try to send to ESP32 (will fail if ESP32 not running)
        print(f"Sending to ESP32...")
        success = send_activity_packet(display)
        if success:
            print("✓ Packet sent successfully!")
        else:
            print("✗ Failed to send packet (ESP32 may not be running)")
        
        print("-" * 50)

if __name__ == "__main__":
    test_activity_packet() 