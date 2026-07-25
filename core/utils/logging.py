import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

# Standard attributes that are always present in LogRecord objects
STANDARD_RECORD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as single-line JSON.
    Automatically captures custom fields passed via the extra dictionary.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Construct the base JSON payload
        log_payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        # Include traceback details if an exception occurred
        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        # Include custom extra variables passed to the log statement
        for key, val in record.__dict__.items():
            if key not in STANDARD_RECORD_ATTRS and not key.startswith("_"):
                log_payload[key] = val

        return json.dumps(log_payload)


def setup_logging(
    level: str = "INFO", log_format: str = "JSON", file_path: Optional[str] = None
) -> None:
    """
    Configures the root logging handler to output standard console
    or structured JSON logs.
    """
    root_logger = logging.getLogger()

    # Remove existing handlers to avoid duplicate log entries
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Choose formatter
    if log_format.upper() == "JSON":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console output handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # File output handler
    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
