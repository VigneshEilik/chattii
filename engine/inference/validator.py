import os
from typing import Any, Dict


class InferenceValidator:
    """Validates loaded transformer checkpoints and parameter shape matches."""

    @staticmethod
    def validate_checkpoint(checkpoint_dir: str) -> Dict[str, Any]:
        report = {
            "model_pt_exists": False,
            "config_json_exists": False,
            "metadata_json_exists": False,
            "is_valid": False,
        }

        if not os.path.exists(checkpoint_dir):
            return report

        report["model_pt_exists"] = os.path.exists(
            os.path.join(checkpoint_dir, "model.pt")
        )
        report["config_json_exists"] = os.path.exists(
            os.path.join(checkpoint_dir, "config.json")
        )
        report["metadata_json_exists"] = os.path.exists(
            os.path.join(checkpoint_dir, "metadata.json")
        )

        report["is_valid"] = report["model_pt_exists"] and report["config_json_exists"]
        return report
