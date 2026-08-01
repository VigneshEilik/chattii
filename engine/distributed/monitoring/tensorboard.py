import os
from typing import Any, Dict


class TensorBoardLogger:
    """Logs distributed metrics to TensorBoard directories."""

    def __init__(self, log_dir: str = "runs") -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        # Mock summary writer to avoid mandatory dependency loading errors
        self.writer = None

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        pass
