import os
from typing import Any, Dict, List, Optional

import yaml


class GenerationConfig:
    """Configuration class for text generation parameters matching HuggingFace API pattern."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        config_dict = config_dict or {}
        self.temperature = float(config_dict.get("temperature", 0.7))
        self.top_k = int(config_dict.get("top_k", 50))
        self.top_p = float(config_dict.get("top_p", 0.9))
        self.min_p = float(config_dict.get("min_p", 0.05))
        self.repetition_penalty = float(config_dict.get("repetition_penalty", 1.1))
        self.frequency_penalty = float(config_dict.get("frequency_penalty", 0.0))
        self.presence_penalty = float(config_dict.get("presence_penalty", 0.0))
        self.max_new_tokens = int(config_dict.get("max_new_tokens", 256))
        self.do_sample = bool(config_dict.get("do_sample", True))
        self.use_cache = bool(config_dict.get("use_cache", True))
        self.seed = int(config_dict.get("seed", 42))
        self.stop_sequences: List[str] = config_dict.get("stop_sequences", [])

    @classmethod
    def load_from_yaml(
        cls, path: str = "configs/generation.yaml"
    ) -> "GenerationConfig":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f) or {}
                return cls(data)
            except Exception:
                return cls()
