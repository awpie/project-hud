# Mock Model for demonstration of modular inference system
import numpy as np
import time
from prediction_result import PredictionResult

class MockModel:
    """Mock model that cycles through activities for demonstration purposes."""
    
    def __init__(self):
        self.activities = ['coding', 'eating', 'reading', 'piano', 'nature', 'idle']
        self.current_index = 0
        self.start_time = time.time()
        self.switch_interval = 3.0  # Switch activity every 3 seconds
        
    def predict(self, frame):
        """Mock prediction that cycles through activities."""
        current_time = time.time()
        
        # Switch activity every few seconds
        if current_time - self.start_time > self.switch_interval:
            self.current_index = (self.current_index + 1) % len(self.activities)
            self.start_time = current_time
        
        # Generate a mock confidence score
        base_confidence = 0.7
        noise = np.random.normal(0, 0.1)  # Add some noise
        confidence = max(0.0, min(1.0, base_confidence + noise))
        
        return PredictionResult(self.activities[self.current_index], confidence)
    
    def cleanup(self):
        """Clean up resources (nothing to do for mock model)."""
        pass 