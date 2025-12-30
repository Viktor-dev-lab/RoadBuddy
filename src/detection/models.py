from detection.multi_model_detector import ModelConfig, MultiModelDetector

detector = MultiModelDetector()

# pothole
detector.register_model(
    "pothole",
    ModelConfig(
        model_id="pothole-detection-lsz3t/3",
        confidence=0.4,
        iou=0.6
    )
)

# traffic sign
detector.register_model(
    "traffic_sign",
    ModelConfig(
        model_id="vietnam-traffic-sign-detection-2i2j8/5",
        confidence=0.5,
        iou=0.6
    )
)
