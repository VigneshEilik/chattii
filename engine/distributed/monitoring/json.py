import json
import os
from typing import Any, Dict


class JSONMetricsLogger:
    """Logs training metrics in JSON-Lines format."""

    def __init__(self, file_path: str = "logs/metrics.jsonl") -> None:
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        row = {"step": step, **metrics}
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
