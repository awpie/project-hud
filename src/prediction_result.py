class PredictionResult:
    """Container for model prediction results with label and confidence."""
    
    def __init__(self, label: str, confidence: float):
        self.label = label
        self.confidence = confidence
    
    def __str__(self):
        return f"PredictionResult(label='{self.label}', confidence={self.confidence:.3f})"
    
    def __repr__(self):
        return self.__str__() 