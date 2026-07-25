import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from engine.transformer.config import ModelMathConfig


class CheckpointLoader:
    """
    Validates file signatures and deserializes model weights and configurations.
    """

    @staticmethod
    def load(
        load_dir: str,
    ) -> Tuple[Dict[str, torch.Tensor], ModelMathConfig, Dict[str, Any]]:
        """
        Validates SHA-256 signatures, returning parameters, configs, and metadata.
        """
        path = Path(load_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint directory '{load_dir}' does not exist."
            )

        # 1. Verify checksum.sha256
        checksum_path = path / "checksum.sha256"
        if not checksum_path.exists():
            raise FileNotFoundError("Missing required checksum.sha256 signature file.")

        with open(checksum_path, "r", encoding="utf-8") as f:
            checksums = json.load(f)

        for name, expected_hash in checksums.items():
            file_path = path / name
            if not file_path.exists():
                raise FileNotFoundError(f"Missing expected checkpoint file: '{name}'")

            h = hashlib.sha256()
            with open(file_path, "rb") as f_bin:
                while chunk := f_bin.read(8192):
                    h.update(chunk)

            if h.hexdigest() != expected_hash:
                raise ValueError(
                    f"Checksum mismatch for file: '{name}'. File may be corrupted."
                )

        # 2. Load config.json
        config_path = path / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        config = ModelMathConfig(**config_data)

        # 3. Load metadata.json
        meta_path = path / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        # 4. Load model.pt
        model_path = path / "model.pt"
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        return state_dict, config, meta_data
