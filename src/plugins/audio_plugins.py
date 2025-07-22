"""
Audio plugins for the HUD inference system.

This module demonstrates how to create new plugin types that can be
automatically registered and used by modules.
"""

import time
import threading
import numpy as np

class AudioPlugin:
    """Base class for audio plugins."""
    
    def __init__(self):
        self.is_initialized = False
        self.sample_rate = 16000
        self.chunk_size = 1024
    
    def initialize(self):
        """Initialize the audio plugin. Override in subclasses."""
        pass
    
    def read_audio(self):
        """Read audio data. Override in subclasses."""
        pass
    
    def is_active(self):
        """Check if audio is active. Override in subclasses."""
        pass
    
    def cleanup(self):
        """Clean up audio resources. Override in subclasses."""
        pass

class MicrophonePlugin(AudioPlugin):
    """Plugin for microphone input."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_thread = None
        self.running = False
        self.current_audio = None
    
    def initialize(self):
        """Initialize microphone."""
        try:
            # This is a placeholder - in a real implementation, you'd initialize
            # the actual microphone hardware here
            print(f"✓ Microphone initialized (sample_rate={self.sample_rate}, chunk_size={self.chunk_size})")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"✗ Failed to initialize microphone: {e}")
            return False
    
    def read_audio(self):
        """Read audio data from microphone."""
        if not self.is_initialized:
            return None
        
        # This is a placeholder - in a real implementation, you'd read
        # actual audio data from the microphone
        # For now, return dummy audio data
        dummy_audio = np.random.randn(self.chunk_size).astype(np.float32)
        return dummy_audio
    
    def is_active(self):
        """Check if microphone is active."""
        return self.is_initialized
    
    def cleanup(self):
        """Clean up microphone resources."""
        self.is_initialized = False
        print("Microphone plugin cleaned up")

class TestAudioPlugin(AudioPlugin):
    """Test audio plugin that generates dummy audio data."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.frame_count = 0
        self.is_initialized = True
    
    def initialize(self):
        """Initialize test audio."""
        print(f"✓ Test audio initialized (sample_rate={self.sample_rate}, chunk_size={self.chunk_size})")
        return True
    
    def read_audio(self):
        """Generate dummy audio data."""
        self.frame_count += 1
        
        # Generate a simple sine wave that changes frequency over time
        t = np.linspace(0, self.chunk_size/self.sample_rate, self.chunk_size)
        frequency = 440 + 100 * np.sin(self.frame_count * 0.1)  # Varying frequency
        audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        return audio_data
    
    def is_active(self):
        """Test audio is always active."""
        return True
    
    def cleanup(self):
        """Clean up test audio."""
        print("Test audio plugin cleaned up")

def create_audio_plugin(audio_type, **kwargs):
    """Factory function to create audio plugins."""
    if audio_type == 'microphone':
        return MicrophonePlugin(**kwargs)
    elif audio_type == 'test':
        return TestAudioPlugin(**kwargs)
    else:
        raise ValueError(f"Unknown audio type: {audio_type}") 