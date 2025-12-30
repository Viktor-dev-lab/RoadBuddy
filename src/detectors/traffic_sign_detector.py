import sys
from pathlib import Path
import supervision as sv
from .base_detector import BaseDetector

ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from detection.models import detector

class TrafficSignDetector(BaseDetector):
    def __init__(self, confidence: float = 0.4):
        super().__init__(name="Traffic Sign", confidence=confidence)
        self.load_model()
    
    def load_model(self):
        self.model = detector.get_model("traffic_sign")
        print(f"✓ {self.name} model loaded!")
    
    def detect(self, image) -> sv.Detections:
        result = self.model.infer(image)[0]
        detections = sv.Detections.from_inference(result)
        return detections