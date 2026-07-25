import os
from typing import List

import yaml
from pydantic import BaseModel, Field


class TokenizerConfig(BaseModel):
    """
    Configuration settings schema for all tokenizer algorithms in Chatii.
    """

    algorithm: str = Field(default="bpe")
    vocab_size: int = Field(default=1000)
    unicode_normalization: str = Field(default="NFKC")
    lowercase: bool = Field(default=False)
    min_frequency: int = Field(default=1)
    special_tokens: List[str] = Field(
        default_factory=lambda: ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
    )
    checkpoint_interval: int = Field(default=100)
    padding: bool = Field(default=True)
    max_length: int = Field(default=4096)

    @classmethod
    def load_from_yaml(cls, filepath: str) -> "TokenizerConfig":
        """
        Parses a YAML configuration file. Falls back to defaults if not found.
        """
        if not os.path.exists(filepath):
            return cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
