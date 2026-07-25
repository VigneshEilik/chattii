from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel, Field


class ModelMathConfig(BaseModel):
    """
    Pydantic schema model loading mathematical layer settings from configs/model.yaml.
    """

    activation: str = Field(default="gelu")
    normalization: str = Field(default="rmsnorm")
    position_embedding: str = Field(default="rope")
    dropout: float = Field(default=0.0)
    initializer: str = Field(default="xavier_uniform")
    dtype: str = Field(default="float32")
    device: str = Field(default="cpu")
    d_model: int = Field(default=512)
    n_heads: int = Field(default=8)
    d_ff: int = Field(default=2048)
    num_layers: int = Field(default=12)
    rope_theta: float = Field(default=10000.0)
    max_seq_len: int = Field(default=2048)
    vocab_size: int = Field(default=32000)

    @classmethod
    def load_from_yaml(
        cls, path: Union[str, Path] = "configs/model.yaml"
    ) -> "ModelMathConfig":
        """Loads model settings from YAML file path."""
        file_path = Path(path)
        if not file_path.exists():
            return cls()
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
