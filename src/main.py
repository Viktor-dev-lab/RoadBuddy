import sys
import cv2
import json
import numpy as np
import supervision as sv
from pathlib import Path

# Thêm ROOT_DIR vào sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Project Imports
from src.config.settings import RTDETR_MODEL, DEVICE, CONFIDENCE_THRESHOLDS
from src.detectors.rtdetr_detector import RTDETRDetector
from src.detectors.traffic_sign_detector import TrafficSignDetector
from src.detectors.pothole_detector import PotholeDetector
from src.utils.visualization import Visualizer
from src.pipeline.detection_pipeline import DetectionPipeline
from src.processing.video_processor import VideoPreprocessor, CropConfigurator
from src.kinematic.KinematicCalculator import KinematicCalculator 
from src.kinematic.KinematicVisualizer import KinematicVisualizer

# ================= CONFIGURATION =================
VIDEO_INPUT = "data/videos/Dashcam2.mp4"
VIDEO_OUTPUT = "output/videos/tracked_dashcam.mp4"
JSON_OUTPUT = "output/data/telemetry_data.json"

FLAGS = {
    "use_rtdetr": True,
    "use_traffic_sign": True,
    "use_pothole": True,
    "enable_tracking": True,
    "enable_kinematic": True,
    "use_segmentation": False,
    "enable_preprocessing": True,
    "auto_configure_crop": True
}

FRAME_STRIDE = 2
PIXELS_PER_METER = 20.0
DYNAMIC_OBJECTS = {"person", "bicycle", "motorcycle", "car", "bus", "truck"}
CROP_SETTINGS = {"bottom": 0.35, "top": 0.0, "left": 0.0, "right": 0.0}

class DashcamProcessor:
    def __init__(self):
        self.video_path = Path(VIDEO_INPUT)
        self.output_path = Path(VIDEO_OUTPUT)
        self.json_path = Path(JSON_OUTPUT)
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.preprocessor = self._init_preprocessor()
        self.detectors = self._init_detectors()
        self.segmentor = self._init_segmentor()
        self.visualizer = Visualizer()
        
        # Dữ liệu lịch sử
        self.object_history = []
        self.kinematic_history = []
        self.lane_history = []
        self.ego_history = [] # Lưu chuyển động xe chủ

    def _init_preprocessor(self):
        if not FLAGS["enable_preprocessing"]: return None
        cfg = CROP_SETTINGS
        if FLAGS["auto_configure_crop"]:
            configurator = CropConfigurator(self.video_path)
            auto = configurator.run()
            cfg = {"bottom": auto['crop_bottom_ratio'], "top": auto['crop_top_ratio'], "left": auto['crop_left_ratio'], "right": auto['crop_right_ratio']}
        return VideoPreprocessor(crop_bottom_ratio=cfg["bottom"], crop_top_ratio=cfg["top"],crop_left_ratio=cfg["left"], crop_right_ratio=cfg["right"])

    def _init_detectors(self):
        dets = []
        if FLAGS["use_rtdetr"]: dets.append(RTDETRDetector(model_path=RTDETR_MODEL, confidence=CONFIDENCE_THRESHOLDS["rtdetr"]))
        if FLAGS["use_traffic_sign"]: dets.append(TrafficSignDetector())
        if FLAGS["use_pothole"]: dets.append(PotholeDetector())
        return dets

    def _init_segmentor(self):
        if not FLAGS["use_segmentation"]: return None
        from src.segmentation.sam2_segmentor import SAM2Segmentor
        from src.config.settings import SAM2_CONFIG, SAM2_CHECKPOINT
        return SAM2Segmentor(config_path=SAM2_CONFIG, checkpoint_path=SAM2_CHECKPOINT, device=DEVICE)

    def run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        origin_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_fps = origin_fps / FRAME_STRIDE 
        
        ret, frame = cap.read()
        if not ret: return
        if self.preprocessor: frame = self.preprocessor.process_frame(frame)
        h, w = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        pipeline = DetectionPipeline(detectors=self.detectors, segmentor=self.segmentor, visualizer=self.visualizer,
                                     enable_tracking=FLAGS["enable_tracking"], enable_kinematic=FLAGS["enable_kinematic"],
                                     pixels_per_meter=PIXELS_PER_METER)

        if FLAGS["enable_kinematic"]:
            pipeline.kinematic_calc = KinematicCalculator(fps=output_fps, pixels_per_meter=PIXELS_PER_METER)
            pipeline.kinematic_viz = KinematicVisualizer()

        out = cv2.VideoWriter(str(self.output_path), cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (w, h))
        
        frame_count = 0
        processed_idx = 0 
        
        try:
            while cap.isOpened():
                for _ in range(FRAME_STRIDE - 1):
                    cap.grab()
                    frame_count += 1
                
                ret, frame = cap.read()
                if not ret: break
                frame_count += 1
                processed_idx += 1 
                curr_time = round(frame_count / origin_fps, 2)
                
                if self.preprocessor: frame = self.preprocessor.process_frame(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Phát hiện & Tracking
                results = [det.detect(frame) for det in self.detectors]
                results = [r for r in results if len(r) > 0]
                detections = sv.Detections.merge(results) if results else sv.Detections.empty()

                if FLAGS["enable_tracking"] and len(detections) > 0:
                    detections = self._handle_tracking_kinematic_sam(pipeline, detections, frame, frame_rgb, processed_idx)
                    
                    # Thu thập dữ liệu xung quanh VÀ xe chủ
                    self._collect_telemetry(detections, frame_count, curr_time, w)
                    self._collect_ego_motion(curr_time)

                annotated = self._draw(pipeline, frame, detections)
                out.write(annotated)

                if processed_idx % 15 == 0:
                    self._log_progress(pipeline, frame_count, total_frames, detections)

        finally:
            cap.release()
            out.release()
            self._save_json_report()
            print(f"Xử lý hoàn tất.")

    def _handle_tracking_kinematic_sam(self, pipeline, detections, frame, frame_rgb, p_idx):
        raw_dets = []
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            raw_dets.append(([x1, y1, x2-x1, y2-y1], detections.confidence[i], detections.data["class_name"][i]))

        tracks = pipeline.tracker.update_tracks(raw_dets, frame=frame_rgb)
        t_boxes, t_confs, t_names, t_ids = [], [], [], []
        for t in tracks:
            if not t.is_confirmed(): continue
            t_boxes.append(t.to_ltrb())
            t_confs.append(t.get_det_conf() or 1.0)
            t_names.append(t.get_det_class() or "unknown")
            t_ids.append(t.track_id)

        if not t_boxes: return sv.Detections.empty()

        tracked_detections = sv.Detections(xyxy=np.array(t_boxes), confidence=np.array(t_confs), 
                                           class_id=np.arange(len(t_boxes)), tracker_id=np.array(t_ids))
        tracked_detections.data["class_name"] = np.array(t_names)

        if FLAGS["enable_kinematic"] and pipeline.kinematic_calc:
            k_states = []
            for i in range(len(tracked_detections)):
                name = tracked_detections.data["class_name"][i].lower()
                if name in DYNAMIC_OBJECTS:
                    state = pipeline.kinematic_calc.update(tracked_detections.tracker_id[i], 
                                                           tracked_detections.xyxy[i], p_idx, 
                                                           name, tracked_detections.confidence[i])
                    k_states.append(state)
                else: k_states.append(None)
            tracked_detections.data["kinematic"] = k_states
        return tracked_detections

    def _collect_ego_motion(self, t_s):
        """Giả lập/Thu thập chuyển động xe chủ (Ego vehicle)"""
        # Ở đây bạn có thể thay bằng logic lấy speed từ cảm biến thực
        mock_ego_speed = 15.5  # m/s (~55 km/h)
        mock_ego_accel = 0.1   # m/s2
        
        self.ego_history.append({
            "time_s": t_s,
            "track_id": 0, # Mặc định ID 0 cho Ego Vehicle
            "speed_mps": mock_ego_speed,
            "accel_mps2": mock_ego_accel,
            "yaw_rate_deg_s": 0.0,
            "method": "visual_odometry_placeholder"
        })

    def _collect_telemetry(self, detections, f_id, t_s, img_w):
        for i in range(len(detections)):
            tid, bbox, cls = int(detections.tracker_id[i]), detections.xyxy[i].tolist(), detections.data["class_name"][i]
            
            # 1. Object Track
            self.object_history.append({
                "time_s": t_s, "frame_id": f_id, "track_id": tid, "class": cls,
                "bbox": [round(x, 1) for x in bbox], "track_quality": round(float(detections.confidence[i]), 2)
            })
            
            # 2. Kinematics (Đã thêm track_id để Join)
            state = detections.data.get("kinematic", [None]*len(detections))[i]
            if state:
                self.kinematic_history.append({
                    "time_s": t_s, 
                    "track_id": tid, 
                    "speed_mps": round(float(state.speed), 2),
                    "accel_mps2": round(float(state.accel), 2), 
                    "yaw_rate_deg_s": 0.0,
                    "confidence": round(float(state.conf), 2)
                })
            
            # 3. Lane assignment
            cx = (bbox[0] + bbox[2]) / 2
            lid = "left_lane" if cx < img_w*0.33 else "right_lane" if cx > img_w*0.66 else "ego_lane"
            self.lane_history.append({"time_s": t_s, "track_id": tid, "lane_id": lid, "lane_conf": 0.90})

    def _draw(self, pipeline, frame, detections):
        annotated = frame.copy()
        if FLAGS["enable_kinematic"] and "kinematic" in detections.data:
            for i in range(len(detections)):
                state = detections.data["kinematic"][i]
                if state:
                    annotated = pipeline.kinematic_viz.draw_metrics(annotated, state, detections.xyxy[i], pipeline.kinematic_calc)
                    annotated = pipeline.kinematic_viz.draw_trajectory(annotated, state)
        return self.visualizer.annotate_with_tracking(annotated, detections)

    def _log_progress(self, pipeline, f_count, total, detections):
        p = (f_count / total) * 100
        print(f"Progress: {f_count}/{total} ({p:.1f}%) | Active Objects: {len(detections)}  | Speed: {pipeline.kinematic_calc.get_ego_speed()} m/s")

    def _save_json_report(self):
        report = {
            "ego_motion": self.ego_history,          # Bảng chuyển động xe chủ
            "object_tracks": self.object_history,    # Bảng vết vật thể
            "kinematics": self.kinematic_history,    # Bảng động học (có track_id)
            "lane_assignments": self.lane_history    # Bảng phân làn
        }
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"📊 JSON report saved to {self.json_path}")

if __name__ == "__main__":
    DashcamProcessor().run()
    #  python .\src\main.py