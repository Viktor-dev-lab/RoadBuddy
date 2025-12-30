from dataclasses import dataclass
from typing import List

@dataclass
class TrackSampleDC:
    time_s: float
    track_id: int
    class_name: str
    bbox: List[float]
    track_quality: float

@dataclass
class EgoMotionSampleDC:
    time_s: float
    speed_mps: float
    accel_mps2: float
    yaw_rate_deg_s: float
    method: str
    confidence: float

@dataclass
class TrackLaneSampleDC:
    time_s: float
    track_id: int
    lane_id: str
    lane_conf: float
