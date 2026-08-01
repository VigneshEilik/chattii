import csv
import os
from typing import Any, Dict


class CSVMetricsLogger:
    """Logs training metrics to tabular CSV files."""

    def __init__(self, file_path: str = "logs/metrics.csv") -> None:
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        row = {"step": step, **metrics}
        file_exists = os.path.exists(self.file_path)
        with open(self.file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
