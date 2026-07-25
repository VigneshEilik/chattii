# ruff: noqa: E501
import json
import os
import subprocess
import time
from typing import Any, Dict, Optional


class ExperimentTracker:
    """Logs run configurations, git states, and metrics for single training experiments."""

    def __init__(self, run_id: str, experiments_root: str = "experiments") -> None:
        self.run_id = run_id
        self.experiments_root = experiments_root
        self.run_dir = os.path.join(self.experiments_root, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.start_time = time.perf_counter()

    def get_git_commit(self) -> str:
        """Retrieves active Git commit hash identifier for traceability."""
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:
            return "unknown"

    def log_metadata(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Saves configuration and execution settings to metadata.json."""
        git_hash = self.get_git_commit()
        info = {
            "run_id": self.run_id,
            "git_commit": git_hash,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": config,
            "metadata": metadata,
        }
        with open(
            os.path.join(self.run_dir, "metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(info, f, indent=2)

    def log_metrics(
        self,
        step: int,
        loss: float,
        lr: float,
        val_loss: Optional[float] = None,
        val_perplexity: Optional[float] = None,
    ) -> None:
        """Logs step-wise training and validation metrics to metrics.csv."""
        metric_file = os.path.join(self.run_dir, "metrics.csv")
        exists = os.path.exists(metric_file)
        with open(metric_file, "a", encoding="utf-8") as f:
            if not exists:
                f.write("step,loss,lr,val_loss,val_perplexity\n")
            f.write(
                f"{step},{loss:.6f},{lr:.2e},"
                f"{val_loss if val_loss is not None else ''},"
                f"{val_perplexity if val_perplexity is not None else ''}\n"
            )
