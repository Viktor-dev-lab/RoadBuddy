import sys
from pathlib import Path
import supervision as sv
from .base_detector import BaseDetector

ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from detection.models import detector

class PotholeDetector(BaseDetector):
    def __init__(self, confidence: float = 0.2):
        super().__init__(name="Pothole", confidence=confidence)
        self.load_model()
    
    def load_model(self):
        self.model = detector.get_model("pothole")
        print(f"✓ {self.name} model loaded!")
    
    def detect(self, image) -> sv.Detections:
        result = self.model.infer(image)[0]
        detections = sv.Detections.from_inference(result)
        return detections