# ruff: noqa: E501
import json
import os
from typing import Any, Dict, List


class ExperimentComparison:
    """Loads and compiles metrics across multiple experiment runs for easy comparison."""

    @staticmethod
    def compare(experiments_root: str = "experiments") -> List[Dict[str, Any]]:
        """
        Loads metadata.json from all run subdirectories and returns comparative lists.
        """
        results = []
        if not os.path.exists(experiments_root):
            return []

        for run_id in os.listdir(experiments_root):
            metadata_path = os.path.join(experiments_root, run_id, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    meta = data.get("metadata", {})
                    cfg = data.get("config", {})
                    opt = cfg.get("optimizer", {})

                    results.append(
                        {
                            "run_id": run_id,
                            "created_at": data.get("created_at"),
                            "best_loss": meta.get("best_loss"),
                            "best_perplexity": meta.get("best_perplexity"),
                            "epochs": cfg.get("epochs"),
                            "batch_size": cfg.get("batch_size"),
                            "lr": opt.get("lr"),
                        }
                    )
                except Exception:
                    pass

        return results
