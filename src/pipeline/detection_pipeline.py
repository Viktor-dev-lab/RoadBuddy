import cv2
import numpy as np
from typing import List
import supervision as sv
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.detectors.base_detector import BaseDetector
from src.segmentation.sam2_segmentor import SAM2Segmentor
from src.utils.visualization import Visualizer
from src.utils.image_utils import ImageLoader
from src.kinematic.KinematicCalculator import KinematicCalculator
from src.kinematic.KinematicVisualizer import KinematicVisualizer
from deep_sort_realtime.deepsort_tracker import DeepSort

class DetectionPipeline:
    def __init__(self, detectors: List[BaseDetector], segmentor: SAM2Segmentor, visualizer: Visualizer, enable_tracking: bool = False, enable_kinematic: bool = False, pixels_per_meter: float = 20.0):
        self.detectors = detectors
        self.segmentor = segmentor
        self.visualizer = visualizer
        self.image_loader = ImageLoader()
        self.enable_tracking = enable_tracking
        self.enable_kinematic = enable_kinematic
        
        if self.enable_tracking:
            self.tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7, max_cosine_distance=0.3, nn_budget=100, embedder="mobilenet", embedder_gpu=True)
        
        # Kinematic calculator
        if self.enable_kinematic:
            self.kinematic_calc = None  
            self.kinematic_viz = KinematicVisualizer()
            self.pixels_per_meter = pixels_per_meter
    
    def process_image(self, image_path: Path, output_path: Path):
        image = self.image_loader.load_image(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run all detectors
        all_detections = []
        detector_stats = {}
        
        for detector in self.detectors:
            detections = detector.detect(image)
            detector_stats[detector.name] = len(detections)
            
            if len(detections) > 0:
                all_detections.append(detections)
        
        # Merge detections
        if all_detections: detections = sv.Detections.merge(all_detections)
        else: detections = sv.Detections.empty()
        
        # Segmentation
        if len(detections) > 0: detections = self.segmentor.segment(image_rgb, detections)
        
        # Visualize
        annotated = self.visualizer.annotate(image, detections)
        
        # Save
        self.image_loader.save_image(annotated, output_path)
        return detections, detector_stats
    