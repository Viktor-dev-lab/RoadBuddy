import numpy as np
import supervision as sv
from ultralytics import RTDETR
from .base_detector import BaseDetector
from src.config.settings import COCO_CLASS_NAMES, ROAD_CLASS_IDS

class RTDETRDetector(BaseDetector):
    def __init__(self, model_path: str = "rtdetr-l.pt", confidence: float = 0.5):
        super().__init__(name="RT-DETR", confidence=confidence)
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        self.model = RTDETR(self.model_path)
        print(f"✓ {self.name} model loaded!")
    
    def detect(self, image) -> sv.Detections:
        results = self.model(image, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter road classes
        detections = detections[detections.confidence >= self.confidence]
        road_mask = np.array([cls_id in ROAD_CLASS_IDS for cls_id in detections.class_id])
        detections = detections[road_mask]
        
        # add class names
        if len(detections) > 0:
            class_names = [COCO_CLASS_NAMES[int(cls_id)] for cls_id in detections.class_id]
            detections.data["class_name"] = np.array(class_names)
        
        return detections