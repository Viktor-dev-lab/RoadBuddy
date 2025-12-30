import numpy as np
from typing import Dict, Optional
from src.kinematic.KinematicState import KinematicState


class KinematicCalculator:

    def __init__(self, fps: float = 30.0, pixels_per_meter: float = 20.0, smooth_window: int = 5, min_history: int = 3,):
        self.fps = fps
        self.dt = 1.0 / fps
        self.ppm = pixels_per_meter
        self.smooth_window = smooth_window
        self.min_history = min_history

        self.states: Dict[int, KinematicState] = {}


    def update(self, track_id: int, bbox: np.ndarray, frame_idx: int, class_name: str = "unknown", confidence: float = 1.0,) -> KinematicState:
        cx, cy = self._bbox_center(bbox)
        ts = frame_idx / self.fps

        # Initialize new track
        if track_id not in self.states:
            state = self._init_state(track_id, cx, cy, ts, class_name, confidence)
            self.states[track_id] = state
            return state

        # Update existing track
        prev = self.states[track_id]
        state = self._compute_state(prev, cx, cy, ts, class_name, confidence)

        # Add to history
        prev.history.append(state)

        # Smooth if enough history
        if len(prev.history) >= self.min_history:
            self._smooth(state, prev.history)

        # Update state
        self.states[track_id] = state
        return state

    def get(self, track_id: int) -> Optional[KinematicState]:
        """Get state for a track"""
        return self.states.get(track_id)
    
    def get_all_states(self) -> Dict[int, KinematicState]:
        """Get all track states"""
        return self.states

    def remove(self, track_id: int):
        """Remove a track"""
        self.states.pop(track_id, None)

    
    def to_kmh(self, speed_mps: float) -> float:
        """Convert m/s to km/h"""
        return speed_mps * 3.6
    
    def get_motion_status(self, state: KinematicState) -> str:
        """Get motion status string"""
        speed_kmh = self.to_kmh(state.speed)
        
        if speed_kmh < 1:
            return "Stopped"
        elif abs(state.accel) < 0.5:
            return "Constant Speed"
        elif state.accel > 0.5:
            return "Accelerating"
        elif state.accel < -0.5:
            return "Braking"
        else:
            return "Moving"
    
    def get_direction(self, state: KinematicState) -> str:
        """Get direction string with arrow"""
        angle = state.yaw_deg
        
        if -22.5 <= angle < 22.5:
            return "East →"
        elif 22.5 <= angle < 67.5:
            return "NE ↗"
        elif 67.5 <= angle < 112.5:
            return "North ↑"
        elif 112.5 <= angle < 157.5:
            return "NW ↖"
        elif angle >= 157.5 or angle < -157.5:
            return "West ←"
        elif -157.5 <= angle < -112.5:
            return "SW ↙"
        elif -112.5 <= angle < -67.5:
            return "South ↓"
        else:
            return "SE ↘"


    def _init_state( self, track_id: int, cx: float, cy: float, ts: float, cls: str, conf: float,) -> KinematicState:
        """Initialize a new track state"""
        return KinematicState(track_id=track_id, timestamp=ts, cx=cx, cy=cy, vx=0.0,vy=0.0, speed=0.0, ax=0.0, ay=0.0, accel=0.0, yaw_deg=0.0, yaw_rate=0.0, cls=cls,conf=conf, )

    def _compute_state( self, prev: KinematicState, cx: float, cy: float, ts: float, cls: str, conf: float,) -> KinematicState:
        # Displacement in meters
        dx_m, dy_m = self._pixel_to_meter(cx - prev.cx, cy - prev.cy)

        # Velocity (m/s)
        vx = dx_m / self.dt
        vy = dy_m / self.dt
        speed = np.hypot(vx, vy)

        # Acceleration (m/s²)
        ax = (vx - prev.vx) / self.dt
        ay = (vy - prev.vy) / self.dt
        accel = np.hypot(ax, ay)

        # Determine if braking (negative acceleration)
        if np.dot([vx, vy], [ax, ay]) < 0: accel = -accel

        # Yaw angle (degrees)
        yaw = np.degrees(np.arctan2(cy - prev.cy, cx - prev.cx))
        
        # Yaw rate (deg/s)
        yaw_rate = self._normalize_angle((yaw - prev.yaw_deg) / self.dt)

        return KinematicState(track_id=prev.track_id, timestamp=ts, cx=cx, cy=cy, vx=vx, vy=vy, speed=speed, ax=ax, ay=ay,accel=accel, yaw_deg=yaw, yaw_rate=yaw_rate, cls=cls, conf=conf,)


    def _bbox_center(self, bbox: np.ndarray) -> tuple:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _pixel_to_meter(self, dx: float, dy: float) -> tuple:
        return dx / self.ppm, dy / self.ppm

    def _smooth(self, state: KinematicState, history):
        """Smooth metrics using moving average"""
        recent = list(history)[-self.smooth_window :]
        
        if len(recent) < 2:
            return
        
        state.speed = np.mean([s.speed for s in recent])
        state.accel = np.mean([s.accel for s in recent])
        state.yaw_rate = np.mean([s.yaw_rate for s in recent])

    @staticmethod
    def _normalize_angle(a: float) -> float:
        """Normalize angle to [-180, 180]"""
        return (a + 180) % 360 - 180