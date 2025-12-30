from dataclasses import dataclass, field
from collections import deque


@dataclass
class KinematicState:
    # Identity
    track_id: int
    timestamp: float
    
    # Position (pixel coordinates)
    cx: float  # center x
    cy: float  # center y
    
    # Velocity (m/s)
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0  # magnitude
    
    # Acceleration (m/s²)
    ax: float = 0.0
    ay: float = 0.0
    accel: float = 0.0  # magnitude (+ accelerating, - braking)
    
    # Rotation
    yaw_deg: float = 0.0      # current angle
    yaw_rate: float = 0.0     # deg/s
    
    # Metadata
    cls: str = "unknown"       # class name
    conf: float = 1.0          # confidence
    
    # History for smoothing (maxlen=10 frames)
    history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def __repr__(self):
        return (
            f"KinematicState(id={self.track_id}, "
            f"speed={self.speed:.2f}m/s, "
            f"accel={self.accel:+.2f}m/s², "
            f"yaw_rate={self.yaw_rate:.1f}°/s)"
        )
    
    def to_dict(self):
        """Convert to dictionary (for JSON export)"""
        return {
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "position": {"x": self.cx, "y": self.cy},
            "velocity": {
                "x": self.vx,
                "y": self.vy,
                "speed_mps": self.speed,
                "speed_kmh": self.speed * 3.6
            },
            "acceleration": {
                "x": self.ax,
                "y": self.ay,
                "accel_mps2": self.accel
            },
            "rotation": {
                "yaw_deg": self.yaw_deg,
                "yaw_rate_deg_s": self.yaw_rate
            },
            "class": self.cls,
            "confidence": self.conf
        }