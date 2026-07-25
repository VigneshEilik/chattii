import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from engine.transformer.config import ModelMathConfig


class CheckpointSaver:
    """
    Saves Transformer parameters and metadata into versioned checksummed directories.
    """

    @staticmethod
    def save(
        save_dir: str,
        state_dict: Dict[str, torch.Tensor],
        config: ModelMathConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Saves weights to model.pt, configs to config.json, and writes checksum.sha256.
        """
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        # 1. Save config.json
        config_path = path / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config.model_dump_json(indent=2))

        # 2. Save metadata.json
        meta_path = path / "metadata.json"
        meta_data = metadata or {}
        meta_data.setdefault("chatii_version", "0.1.0")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

        # 3. Save model.pt
        model_path = path / "model.pt"
        torch.save(state_dict, model_path)

        # 4. Generate checksum.sha256
        checksums = {}
        for name in ["config.json", "metadata.json", "model.pt"]:
            file_path = path / name
            h = hashlib.sha256()
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    h.update(chunk)
            checksums[name] = h.hexdigest()

        checksum_path = path / "checksum.sha256"
        with open(checksum_path, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)
