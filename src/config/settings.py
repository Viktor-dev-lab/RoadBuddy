import os
from pathlib import Path

# PROJECT PATHS
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
IMAGE_DIR = DATA_DIR / "images"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = PROJECT_ROOT / "models"
SAM2_DIR = PROJECT_ROOT / "third_party" / "SAM2"

# CREATE FOLDERS IF NOT EXIST
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# SAM2 CONFIG =
SAM2_CONFIG = SAM2_DIR / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml"
SAM2_CHECKPOINT = SAM2_DIR / "checkpoints" / "sam2.1_hiera_tiny.pt"

# MODEL CONFIG 
RTDETR_MODEL = "rtdetr-l.pt"
DEVICE = "cpu"  # or "cuda"

# DETECTION CONFIG 
CONFIDENCE_THRESHOLDS = {
    "rtdetr": 0.55,
    "traffic_sign": 0.5,
    "pothole": 0.4,
    "lane": 0.4
}

# COCO CLASSES 
COCO_CLASS_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
    'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# ROAD CLASSES FILTER
ROAD_CLASSES = {"person","bicycle","motorcycle","car","bus","truck","traffic light","stop sign"}
ROAD_CLASS_IDS = {i for i, name in enumerate(COCO_CLASS_NAMES) if name in ROAD_CLASSES}

# VISUALIZATION
COLOR_PALETTE = ["#ffff00", "#ff9b00", "#00ff00", "#00ffff"]

# ENVIRONMENT 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"