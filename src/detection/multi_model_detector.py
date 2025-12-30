from dataclasses import dataclass
from typing import Dict, Any
from inference import get_model

@dataclass
class ModelConfig:
    model_id: str
    confidence: float = 0.25
    iou: float = 0.5


class MultiModelDetector:

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._configs: Dict[str, ModelConfig] = {}

    def register_model(self, name: str, config: ModelConfig):
        self._configs[name] = config

    def load_model(self, name: str):
        
        if name in self._models: return self._models[name]
        if name not in self._configs: raise ValueError(f"Model '{name}' is not registered")

        cfg = self._configs[name]

        model = get_model(model_id=cfg.model_id, confidence=cfg.confidence, iou_threshold=cfg.iou)
        self._models[name] = model
        
        return model

    def get_model(self, name: str):
        return self.load_model(name)

    def available_models(self):
        return list(self._configs.keys())
