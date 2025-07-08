#!/usr/bin/env python3
"""
Test script for bitmap generation and sending functionality.
This script demonstrates how to create HUD images and send them to ESP32.
"""

import time
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from activity_display import ActivityDisplay
from esp32_client import (
    create_hud_image, 
    send_bitmap_image, 
    send_bitmap_image_async,
    save_hud_image,
    print_bitmap_info
)

def test_bitmap_generation():
    """Test bitmap generation with various activities."""
    print("=== Bitmap Generation Test ===")
    
    # Create activity display
    display = ActivityDisplay()
    
    # Test activities with different durations and XP levels
    test_scenarios = [
        ("coding", 0.95, 5),      # High confidence coding
        ("eating", 0.78, 3),      # Medium confidence eating
        ("reading", 0.88, 8),     # High confidence reading
        ("piano", 0.65, 2),       # Lower confidence piano
        ("nature", 0.45, 1),      # Low confidence nature
        ("idle", 0.23, 10),       # Very low confidence idle
    ]
    
    for activity, confidence, duration in test_scenarios:
        print(f"\n--- Testing: {activity} (confidence: {confidence:.2f}) ---")
        
        # Update display with activity
        display.update(activity, confidence)
        
        # Simulate activity duration
        time.sleep(duration)
        
        # Generate and save HUD image
        timestamp = time.strftime("%H%M%S")
        filename = f"hud_{activity}_{timestamp}.png"
        save_hud_image(display, filename)
        
        # Print bitmap information
        print_bitmap_info(display)
        
        # Try to send to ESP32
        print("Sending bitmap to ESP32...")
        success = send_bitmap_image(display)
        
        if success:
            print("✓ Bitmap sent successfully!")
        else:
            print("✗ Failed to send bitmap (ESP32 may not be running)")
        
        print("-" * 50)

def test_async_bitmap_sending():
    """Test asynchronous bitmap sending."""
    print("\n=== Async Bitmap Sending Test ===")
    
    display = ActivityDisplay()
    
    # Simulate rapid activity changes
    activities = ["coding", "reading", "piano", "eating", "nature"]
    
    for i, activity in enumerate(activities):
        confidence = 0.7 + (i * 0.05)  # Varying confidence
        display.update(activity, confidence)
        
        print(f"Sending async bitmap for {activity}...")
        send_bitmap_image_async(display)
        
        time.sleep(0.5)  # Short delay between sends
    
    # Wait for async operations to complete
    time.sleep(2)
    print("Async sending test completed!")

def test_xp_progression():
    """Test HUD display with XP progression."""
    print("\n=== XP Progression Test ===")
    
    display = ActivityDisplay()
    
    # Simulate long coding session with XP gain
    print("Simulating a long coding session...")
    
    for minute in range(10):  # 10 minutes of coding
        display.update("coding", 0.9)
        time.sleep(1)  # 1 second per minute for testing
        
        if minute % 2 == 0:  # Every 2 minutes
            timestamp = time.strftime("%H%M%S")
            filename = f"coding_session_{minute:02d}min_{timestamp}.png"
            save_hud_image(display, filename)
            
            print(f"Minute {minute}: XP = {display.get_activity_tracker().get_current_xp()}")
    
    print("XP progression test completed!")

def main():
    """Main test function."""
    print("Bitmap Generation and ESP32 Communication Test")
    print("=" * 50)
    
    try:
        # Test basic bitmap generation
        test_bitmap_generation()
        
        # Test async sending
        test_async_bitmap_sending()
        
        # Test XP progression
        test_xp_progression()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("Check the generated PNG files to see the HUD images.")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 