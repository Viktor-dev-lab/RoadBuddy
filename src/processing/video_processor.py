"""
VideoPreprocessor - Xử lý video dashcam trước khi detection
Các chức năng:
- Crop mui xe phía dưới
- Resize video
- Denoise
- Stabilization
- ROI selection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import json


class VideoPreprocessor:
    """
    Preprocessing cho dashcam video
    
    Features:
    - Auto detect và crop hood (mui xe)
    - Manual crop với ROI
    - Resize để tăng tốc độ
    - Denoise và stabilization
    """
    
    def __init__(self, 
                 crop_bottom_ratio: float = 0.3,
                 crop_top_ratio: float = 0.0,
                 crop_left_ratio: float = 0.0,
                 crop_right_ratio: float = 0.0,
                 resize_width: Optional[int] = None,
                 denoise: bool = False,
                 stabilize: bool = False):
        """
        Args:
            crop_bottom_ratio: Tỉ lệ crop phía dưới (0.3 = cắt 30% dưới)
            crop_top_ratio: Tỉ lệ crop phía trên
            crop_left_ratio: Tỉ lệ crop bên trái
            crop_right_ratio: Tỉ lệ crop bên phải
            resize_width: Width mới sau khi resize (None = giữ nguyên)
            denoise: Có denoise không
            stabilize: Có stabilize không
        """
        self.crop_bottom = crop_bottom_ratio
        self.crop_top = crop_top_ratio
        self.crop_left = crop_left_ratio
        self.crop_right = crop_right_ratio
        self.resize_width = resize_width
        self.denoise = denoise
        self.stabilize = stabilize
        
        # For stabilization
        self.prev_gray = None
        self.transforms = []
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process một frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Processed frame
        """
        original_frame = frame.copy()
        
        # 1. Crop (loại bỏ mui xe)
        frame = self._crop_frame(frame)
        
        # 2. Denoise
        if self.denoise:
            frame = self._denoise_frame(frame)
        
        # 3. Stabilization
        if self.stabilize:
            frame = self._stabilize_frame(frame)
        
        # 4. Resize
        if self.resize_width:
            frame = self._resize_frame(frame)
        
        return frame
    
    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame để loại bỏ mui xe và vùng không cần thiết"""
        h, w = frame.shape[:2]
        
        # Tính toán vùng crop
        top = int(h * self.crop_top)
        bottom = h - int(h * self.crop_bottom)
        left = int(w * self.crop_left)
        right = w - int(w * self.crop_right)
        
        return frame[top:bottom, left:right]
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame để tăng tốc độ xử lý"""
        h, w = frame.shape[:2]
        new_height = int(h * (self.resize_width / w))
        return cv2.resize(frame, (self.resize_width, new_height))
    
    def _denoise_frame(self, frame: np.ndarray) -> np.ndarray:
        """Denoise frame"""
        return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    
    def _stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Stabilize frame (simple method)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame
        
        # Detect feature points
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30
        )
        
        if prev_pts is None:
            self.prev_gray = gray
            return frame
        
        # Calculate optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None
        )
        
        # Filter valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        # Estimate transform
        if len(prev_pts) < 5:
            self.prev_gray = gray
            return frame
        
        transform = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        
        if transform is None:
            self.prev_gray = gray
            return frame
        
        # Apply transform
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(frame, transform, (w, h))
        
        self.prev_gray = gray
        return stabilized
    
    def visualize_crop_area(self, frame: np.ndarray) -> np.ndarray:
        """
        Visualize vùng sẽ bị crop (để debug/configure)
        
        Args:
            frame: Input frame
            
        Returns:
            Frame với visualization
        """
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Tính toán boundaries
        top = int(h * self.crop_top)
        bottom = h - int(h * self.crop_bottom)
        left = int(w * self.crop_left)
        right = w - int(w * self.crop_right)
        
        # Vẽ vùng giữ lại (màu xanh)
        cv2.rectangle(vis, (left, top), (right, bottom), (0, 255, 0), 3)
        
        # Làm tối vùng bị crop
        overlay = vis.copy()
        
        # Top area
        if top > 0:
            cv2.rectangle(overlay, (0, 0), (w, top), (0, 0, 0), -1)
        
        # Bottom area (mui xe)
        if bottom < h:
            cv2.rectangle(overlay, (0, bottom), (w, h), (0, 0, 0), -1)
        
        # Left area
        if left > 0:
            cv2.rectangle(overlay, (0, 0), (left, h), (0, 0, 0), -1)
        
        # Right area
        if right < w:
            cv2.rectangle(overlay, (right, 0), (w, h), (0, 0, 0), -1)
        
        # Blend
        vis = cv2.addWeighted(overlay, 0.5, vis, 0.5, 0)
        
        # Add text
        cv2.putText(vis, "GREEN AREA = KEEP", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, "DARK AREA = CROP", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis, f"Bottom Crop: {self.crop_bottom*100:.0f}%", (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return vis
    
    def auto_detect_hood(self, video_path: Path, num_samples: int = 30) -> float:
        """
        Auto detect hood (mui xe) và suggest crop ratio
        
        Args:
            video_path: Path to video
            num_samples: Số frames để sample
            
        Returns:
            Suggested crop_bottom_ratio
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly
        sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        
        bottom_edges = []
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Focus on bottom half
            h = edges.shape[0]
            bottom_half = edges[h//2:, :]
            
            # Find horizontal edges (likely hood boundary)
            horizontal = cv2.morphologyEx(
                bottom_half,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            )
            
            # Find highest edge in bottom half
            rows_with_edges = np.where(np.sum(horizontal, axis=1) > 100)[0]
            
            if len(rows_with_edges) > 0:
                # Position relative to full frame
                edge_pos = (h//2 + rows_with_edges[0]) / h
                bottom_edges.append(1.0 - edge_pos)
        
        cap.release()
        
        if not bottom_edges:
            print("⚠️  Could not auto-detect hood, using default 0.3")
            return 0.3
        
        # Use median to be robust to outliers
        suggested_ratio = np.median(bottom_edges)
        
        # Add small margin
        suggested_ratio = min(suggested_ratio + 0.05, 0.5)
        
        print(f"✅ Auto-detected hood: crop bottom {suggested_ratio*100:.0f}%")
        
        return suggested_ratio
    
    def save_config(self, config_path: Path):
        """Save preprocessing config to JSON"""
        config = {
            'crop_bottom': self.crop_bottom,
            'crop_top': self.crop_top,
            'crop_left': self.crop_left,
            'crop_right': self.crop_right,
            'resize_width': self.resize_width,
            'denoise': self.denoise,
            'stabilize': self.stabilize
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Config saved to {config_path}")
    
    @classmethod
    def load_config(cls, config_path: Path) -> 'VideoPreprocessor':
        """Load preprocessing config from JSON"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(**config)


# ============================================================
# INTERACTIVE CROP CONFIGURATOR
# ============================================================

class CropConfigurator:
    """Interactive tool để configure crop parameters"""
    
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        
        # Read first frame
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError(f"Cannot read video: {video_path}")
        
        self.h, self.w = self.frame.shape[:2]
        
        # Default values
        self.crop_bottom = 0.3
        self.crop_top = 0.0
        self.crop_left = 0.0
        self.crop_right = 0.0
        
        # Window name
        self.window_name = "Crop Configurator"
    
    def run(self) -> dict:
        """
        Run interactive configurator
        
        Returns:
            Dict with crop parameters
        """
        cv2.namedWindow(self.window_name)
        
        # Create trackbars
        cv2.createTrackbar('Bottom %', self.window_name, 
                          int(self.crop_bottom * 100), 50, self._on_bottom_change)
        cv2.createTrackbar('Top %', self.window_name, 
                          int(self.crop_top * 100), 50, self._on_top_change)
        cv2.createTrackbar('Left %', self.window_name, 
                          int(self.crop_left * 100), 50, self._on_left_change)
        cv2.createTrackbar('Right %', self.window_name, 
                          int(self.crop_right * 100), 50, self._on_right_change)
        
        print("\n" + "=" * 60)
        print("CROP CONFIGURATOR")
        print("=" * 60)
        print("Use sliders to adjust crop areas")
        print("Press 's' to save and exit")
        print("Press 'q' to quit without saving")
        print("=" * 60)
        
        while True:
            vis = self._visualize()
            cv2.imshow(self.window_name, vis)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                # Save and exit
                print("\n✅ Crop parameters saved!")
                break
            elif key == ord('q'):
                print("\n❌ Cancelled")
                self.crop_bottom = 0.3
                break
        
        cv2.destroyAllWindows()
        self.cap.release()
        
        return {
            'crop_bottom_ratio': self.crop_bottom,
            'crop_top_ratio': self.crop_top,
            'crop_left_ratio': self.crop_left,
            'crop_right_ratio': self.crop_right
        }
    
    def _on_bottom_change(self, val):
        self.crop_bottom = val / 100.0
    
    def _on_top_change(self, val):
        self.crop_top = val / 100.0
    
    def _on_left_change(self, val):
        self.crop_left = val / 100.0
    
    def _on_right_change(self, val):
        self.crop_right = val / 100.0
    
    def _visualize(self) -> np.ndarray:
        """Visualize current crop settings"""
        vis = self.frame.copy()
        
        # Calculate boundaries
        top = int(self.h * self.crop_top)
        bottom = self.h - int(self.h * self.crop_bottom)
        left = int(self.w * self.crop_left)
        right = self.w - int(self.w * self.crop_right)
        
        # Draw keep area
        cv2.rectangle(vis, (left, top), (right, bottom), (0, 255, 0), 3)
        
        # Darken crop areas
        overlay = vis.copy()
        
        if top > 0:
            cv2.rectangle(overlay, (0, 0), (self.w, top), (0, 0, 0), -1)
        if bottom < self.h:
            cv2.rectangle(overlay, (0, bottom), (self.w, self.h), (0, 0, 0), -1)
        if left > 0:
            cv2.rectangle(overlay, (0, 0), (left, self.h), (0, 0, 0), -1)
        if right < self.w:
            cv2.rectangle(overlay, (right, 0), (self.w, self.h), (0, 0, 0), -1)
        
        vis = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)
        
        # Add text
        cv2.putText(vis, "GREEN = KEEP | DARK = CROP", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Output: {right-left}x{bottom-top}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis