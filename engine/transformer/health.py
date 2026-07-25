from pathlib import Path
from typing import List, Tuple

from engine.transformer.checkpoint.load import CheckpointLoader


class TransformerHealthChecker:
    """
    Checks checksum integrity and verifies saved checkpoint files formats.
    """

    @staticmethod
    def check_health(load_dir: str) -> Tuple[bool, List[str]]:
        """
        Validates model directory files presence and checksum signatures.
        """
        errors = []
        path = Path(load_dir)
        if not path.exists():
            errors.append(f"Directory '{load_dir}' does not exist.")
            return False, errors

        for name in ["config.json", "metadata.json", "model.pt", "checksum.sha256"]:
            if not (path / name).exists():
                errors.append(f"Missing required model checkpoint file: '{name}'")

        if len(errors) > 0:
            return False, errors

        try:
            _, _, _ = CheckpointLoader.load(load_dir)
        except Exception as e:
            errors.append(f"Checkpoint integrity validation failed: {e}")

        return len(errors) == 0, errors
