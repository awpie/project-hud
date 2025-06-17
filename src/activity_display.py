import cv2
import numpy as np
import time
from datetime import timedelta

class ActivityTracker:
    def __init__(self):
        self.current_activity = None
        self.start_time = None
        self.confidence = 0.0
        self.activity_history = []
        self.base_xp = 0  # Base XP from completed activities
        self.xp_per_activity = 60  # XP gained per minute of activity
        
        # Activity switching parameters
        self.min_activity_duration = 3.0  # Minimum seconds before allowing activity change
        self.confidence_threshold = 0.2  # Minimum confidence to consider an activity
        self.hysteresis_threshold = 0.0   # How much more confident new activity needs to be
        self.switch_confirmation_time = 3.0  # How long new activity must be detected before switching
        self.last_switch_time = 0
        
        # Switch confirmation tracking
        self.pending_activity = None
        self.pending_confidence = 0.0
        self.pending_start_time = None
        
    def update_activity(self, activity, confidence):
        current_time = time.time()
        
        # First, check if we should switch activities
        should_switch = False
        
        # Case 1: No current activity, and new activity meets confidence threshold
        if self.current_activity is None and confidence >= self.confidence_threshold:
            should_switch = True
            
        # Case 2: Have current activity, and new activity is different and better
        elif (self.current_activity is not None and 
              activity != self.current_activity and
              confidence >= self.confidence_threshold and
              current_time - self.last_switch_time >= self.min_activity_duration):
            # If hysteresis is enabled, check if new activity is significantly better
            if self.hysteresis_threshold > 0:
                if confidence >= self.confidence + self.hysteresis_threshold:
                    should_switch = True
            else:
                should_switch = True
        
        # Handle switch confirmation
        if should_switch:
            if self.pending_activity != activity:
                # New pending activity, start tracking it
                self.pending_activity = activity
                self.pending_confidence = confidence
                self.pending_start_time = current_time
            elif current_time - self.pending_start_time >= self.switch_confirmation_time:
                # Pending activity has been consistent long enough, perform the switch
                if self.current_activity is not None:
                    # Save previous activity duration and add to base XP
                    duration = time.time() - self.start_time
                    xp_gained = int(duration / 60 * self.xp_per_activity)
                    self.activity_history.append({
                        'activity': self.current_activity,
                        'duration': duration,
                        'xp_gained': xp_gained
                    })
                    self.base_xp += xp_gained
                
                self.current_activity = activity
                self.start_time = time.time()
                self.last_switch_time = current_time
                self.pending_activity = None  # Reset pending activity
        else:
            # Reset pending activity if conditions are no longer met
            self.pending_activity = None
            
        # Always update confidence for current activity
        self.confidence = confidence
    
    def get_current_duration(self):
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def get_current_xp(self):
        # Calculate XP from current activity
        current_xp = self.base_xp
        if self.start_time is not None:
            current_duration = self.get_current_duration()
            current_xp += int(current_duration / 60 * self.xp_per_activity)
        return current_xp
    
    def get_xp_progress(self):
        return self.get_current_xp() % 100  # Show progress to next level (0-99)

    def get_activity_summary(self):
        summary = []
        for activity in self.activity_history:
            summary.append(f"{activity['activity']}: {timedelta(seconds=int(activity['duration']))} (XP: +{activity['xp_gained']})")
        summary.append(f"Total XP: {self.get_current_xp()}")
        return summary

class ActivityDisplay:
    def __init__(self):
        self.activity_tracker = ActivityTracker()
        # Create OLED simulation window
        cv2.namedWindow("OLED Simulation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("OLED Simulation", 256, 128)  # 2x scale for better visibility
        
    def update(self, activity, confidence):
        """Update the activity tracker with new activity and confidence"""
        self.activity_tracker.update_activity(activity, confidence)
        self._update_display()
        
    def _update_display(self):
        """Update the OLED display with current activity information"""
        # Create a black canvas for OLED simulation
        oled = np.zeros((64, 128), dtype=np.uint8)
        
        # Draw activity name and confidence
        activity_text = f"{self.activity_tracker.current_activity or 'No Activity'}"
        confidence_text = f"{self.activity_tracker.confidence:.1%}"
        cv2.putText(oled, activity_text, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1)
        cv2.putText(oled, confidence_text, (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1)
        
        # Draw pending activity if there is one
        if self.activity_tracker.pending_activity is not None:
            pending_time = time.time() - self.activity_tracker.pending_start_time
            pending_text = f"-> {self.activity_tracker.pending_activity} ({pending_time:.1f}s)"
            cv2.putText(oled, pending_text, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1)
        
        # Draw XP bar
        xp_progress = self.activity_tracker.get_xp_progress()
        cv2.rectangle(oled, (2, 30), (126, 38), 255, 1)  # XP bar border
        cv2.rectangle(oled, (2, 30), (2 + int(124 * xp_progress / 100), 38), 255, -1)  # XP progress
        
        # Draw timer
        duration = self.activity_tracker.get_current_duration()
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        timer_text = f"{minutes:02d}:{seconds:02d}"
        cv2.putText(oled, timer_text, (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        
        # Draw total XP
        xp_text = f"XP: {self.activity_tracker.get_current_xp()}"
        cv2.putText(oled, xp_text, (2, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1)
        
        cv2.imshow("OLED Simulation", oled)
        
    def cleanup(self):
        """Clean up resources and print activity summary"""
        cv2.destroyWindow("OLED Simulation")
        print("\nActivity Summary:")
        for line in self.activity_tracker.get_activity_summary():
            print(line) 