# Activity to integer mapping for ESP32 communication

ACTIVITY_TO_INT = {
    'coding': 0,
    'eating': 1,
    'reading': 2,
    'piano': 3,
    'nature': 4,
    'idle': 5
}

def activity_to_int(activity: str) -> int:
    """Convert activity label to integer for ESP32.
    
    Args:
        activity (str): Activity label (e.g., 'coding', 'eating', etc.)
        
    Returns:
        int: Integer representation of the activity
    """
    return ACTIVITY_TO_INT.get(activity, 5)  # Default to 'idle' (5) if unknown 