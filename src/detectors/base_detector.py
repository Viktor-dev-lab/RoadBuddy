from abc import ABC, abstractmethod
import supervision as sv

class BaseDetector(ABC):   
    def __init__(self, name: str, confidence: float = 0.5):
        self.name = name
        self.confidence = confidence
        self.model = None
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def detect(self, image) -> sv.Detections:
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, confidence={self.confidence})"