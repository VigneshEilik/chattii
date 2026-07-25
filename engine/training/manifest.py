import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union


class TrainingManifest:
    """Generates manifest JSON documents and validates dataset fingerprints."""

    @staticmethod
    def generate_fingerprint(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Computes SHA-256 hash and size metadata of the dataset file.
        """
        path = Path(file_path)
        if not path.exists():
            return {}

        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)

        return {
            "dataset_name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": h.hexdigest(),
        }

    @staticmethod
    def write_manifest(
        save_path: Union[str, Path],
        model_name: str,
        dataset_fingerprint: Dict[str, Any],
        tokens_seen: int,
        epochs: int,
        checkpoint: str,
        status: str = "completed",
    ) -> None:
        """Writes training metadata manifest files in JSON format."""
        manifest = {
            "model": model_name,
            "dataset": dataset_fingerprint,
            "tokens": tokens_seen,
            "epochs": epochs,
            "checkpoint": checkpoint,
            "status": status,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
