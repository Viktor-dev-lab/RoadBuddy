import cv2
import numpy as np
import torch
import supervision as sv
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Segmentor:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cpu"):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.predictor = None
        self.load_model()
    
    def load_model(self):
        sam2_model = build_sam2(str(self.config_path), str(self.checkpoint_path), device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print("✓ SAM2 model loaded!")
    
    def segment(self, image_rgb: np.ndarray, detections: sv.Detections) -> sv.Detections:
        
        if len(detections) == 0: return detections
        
        self.predictor.set_image(image_rgb)
        masks, scores, _ = self.predictor.predict(box=detections.xyxy, multimask_output=False)
        detections.mask = masks.squeeze(1) > 0
        
        return detections