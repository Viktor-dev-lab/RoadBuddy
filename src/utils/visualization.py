import supervision as sv
import numpy as np
from src.config.settings import COLOR_PALETTE

class Visualizer:
    """Visualization utilities with tracking support"""
    
    def __init__(self):
        color = sv.ColorPalette.from_hex(COLOR_PALETTE)
        self.box_annotator = sv.BoxAnnotator(color=color, thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            color=color, 
            text_color=sv.Color.BLACK,
            text_scale=0.5,
            text_thickness=1
        )
        self.mask_annotator = sv.MaskAnnotator(color=color)
        
        # Trace annotator để vẽ đường đi của object
        self.trace_annotator = sv.TraceAnnotator(
            color=color,
            position=sv.Position.CENTER,
            trace_length=30,
            thickness=2
        )
    
    def annotate(self, image, detections: sv.Detections):
        """Vẽ boxes, masks, labels (cho ảnh)"""
        if len(detections) == 0:
            return image
        
        annotated = self.mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
        
        # Tạo labels
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(detections.data["class_name"], detections.confidence)
        ]
        
        annotated = self.label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels
        )
        
        return annotated
    
    def annotate_with_tracking(self, image, detections: sv.Detections):
        """Vẽ boxes, labels với tracking IDs (cho video)"""
        if len(detections) == 0:
            return image
        
        annotated = image.copy()
        
        # Vẽ trace (đường đi) nếu có tracker_id
        if detections.tracker_id is not None:
            annotated = self.trace_annotator.annotate(
                scene=annotated,
                detections=detections
            )
        
        # Vẽ boxes
        annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
        
        # Tạo labels với tracking ID
        labels = []
        for i in range(len(detections)):
            class_name = detections.data["class_name"][i]
            confidence = detections.confidence[i]
            
            if detections.tracker_id is not None:
                track_id = detections.tracker_id[i]
                label = f"ID:{track_id} {class_name} {confidence:.2f}"
            else:
                label = f"{class_name} {confidence:.2f}"
            
            labels.append(label)
        
        annotated = self.label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels
        )
        
        return annotated