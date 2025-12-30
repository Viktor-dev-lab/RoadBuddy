import cv2
import numpy as np
from typing import Tuple
from src.kinematic.KinematicState import KinematicState


class KinematicVisualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        self.line_thickness = 2
    
    def draw_metrics(self, frame: np.ndarray, state: KinematicState, bbox: np.ndarray, calculator,) -> np.ndarray:
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Get metrics
        speed_kmh = calculator.to_kmh(state.speed)
        motion_status = calculator.get_motion_status(state)
        direction = calculator.get_direction(state)
        
        # Prepare text lines
        texts = [
            f"ID: {state.track_id} | {state.cls}",
            f"Speed: {speed_kmh:.1f} km/h",
            f"Accel: {state.accel:+.2f} m/s²",
            f"Yaw: {state.yaw_rate:.1f}°/s",
            f"Status: {motion_status}",
            f"Dir: {direction}"
        ]
        
        # Calculate background box size
        text_height = 22
        box_height = len(texts) * text_height + 10
        box_width = x2 - x1
        
        if y1 - box_height < 0:
            box_y1 = y2
            box_y2 = y2 + box_height
        else:
            box_y1 = y1 - box_height
            box_y2 = y1
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, box_y1), (x2, box_y2), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw texts
        for i, text in enumerate(texts):
            y_pos = box_y1 + (i + 1) * text_height
            
            # Choose color based on status
            if "Braking" in motion_status: color = (0, 0, 255)  # Red
            elif "Accelerating" in motion_status: color = (0, 255, 0)  # Green
            elif "Stopped" in motion_status: color = (128, 128, 128)  # Gray
            else: color = (255, 255, 255)  # White
            
            cv2.putText(frame, text, (x1 + 5, y_pos), self.font, self.font_scale, color, self.font_thickness, cv2.LINE_AA)
        
        # Draw velocity vector
        self._draw_velocity_vector(frame, state)
        
        return frame
    
    def _draw_velocity_vector(self, frame: np.ndarray, state: KinematicState, scale: float = 5.0):
        center_x = int(state.cx)
        center_y = int(state.cy)
        
        # Scale velocity for visualization
        vx = int(state.vx * scale)
        vy = int(state.vy * scale)
        
        # Only draw if moving
        if abs(vx) > 1 or abs(vy) > 1:
            cv2.arrowedLine(frame, (center_x, center_y), (center_x + vx, center_y + vy), (0, 255, 255), self.line_thickness, tipLength=0.3)
    
    def draw_trajectory(self, frame: np.ndarray, state: KinematicState, color: Tuple[int, int, int] = (255, 0, 255), thickness: int = 2) -> np.ndarray:
        if len(state.history) < 2: return frame
        
        # Get points from history
        points = [(int(s.cx), int(s.cy)) for s in state.history]
        
        # Draw lines between consecutive points
        for i in range(len(points) - 1):
            # Fade effect: older points are more transparent
            alpha = (i + 1) / len(points)
            
            # Create faded color
            faded_color = tuple(int(c * alpha) for c in color)
            
            cv2.line(frame, points[i], points[i + 1], faded_color, thickness, cv2.LINE_AA)
        
        return frame
    
    def draw_speedometer(self, frame: np.ndarray, state: KinematicState, calculator, position: Tuple[int, int] = (10, 50)) -> np.ndarray:
        x, y = position
        speed_kmh = calculator.to_kmh(state.speed)
        
        # Draw background
        cv2.rectangle(frame, (x, y),(x + 200, y + 80), (0, 0, 0), -1)        
        # Draw speed
        cv2.putText(frame, f"{speed_kmh:.1f}", (x + 10, y + 50), cv2.FONT_HERSHEY_BOLD, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "km/h", (x + 120, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,cv2.LINE_AA)
        
        return frame