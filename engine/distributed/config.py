import os
from typing import Any, Dict, Optional

import yaml


class DistributedConfig:
    """Manages distributed training configurations."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        config_dict = config_dict or {}
        self.backend = config_dict.get("backend", "gloo")
        self.strategy = config_dict.get("strategy", "ddp")
        self.world_size = int(config_dict.get("world_size", 1))
        self.mixed_precision = config_dict.get("mixed_precision", "bf16")
        self.compile = bool(config_dict.get("compile", False))
        self.gradient_checkpointing = bool(
            config_dict.get("gradient_checkpointing", False)
        )
        self.flash_attention = config_dict.get("flash_attention", "auto")
        self.checkpoint_interval = int(config_dict.get("checkpoint_interval", 1000))
        self.tensorboard = bool(config_dict.get("tensorboard", True))
        self.csv_logging = bool(config_dict.get("csv_logging", True))
        self.master_addr = config_dict.get("master_addr", "127.0.0.1")
        self.master_port = int(config_dict.get("master_port", 29500))

    @classmethod
    def load_from_yaml(
        cls, path: str = "configs/distributed.yaml"
    ) -> "DistributedConfig":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f) or {}
                return cls(data)
            except Exception:
                return cls()
