import os
import shutil
import tempfile

from engine.distributed.monitoring.csv import CSVMetricsLogger
from engine.distributed.monitoring.json import JSONMetricsLogger
from engine.distributed.monitoring.metrics import TrainingMetricsTracker


def test_monitoring_trackers_and_loggers() -> None:
    tracker = TrainingMetricsTracker()
    stats = tracker.step(loss=1.23, lr=1e-4, tokens=512, examples=4)

    assert stats["loss"] == 1.23
    assert stats["learning_rate"] == 1e-4
    assert stats["tokens_per_sec"] > 0
    assert stats["elapsed_seconds"] >= 0

    temp_dir = tempfile.mkdtemp()
    try:
        # Test CSV logger
        csv_file = os.path.join(temp_dir, "metrics.csv")
        csv_logger = CSVMetricsLogger(csv_file)
        csv_logger.log_metrics(step=1, metrics=stats)
        assert os.path.exists(csv_file)

        # Test JSON logger
        jsonl_file = os.path.join(temp_dir, "metrics.jsonl")
        json_logger = JSONMetricsLogger(jsonl_file)
        json_logger.log_metrics(step=1, metrics=stats)
        assert os.path.exists(jsonl_file)
    finally:
        shutil.rmtree(temp_dir)
