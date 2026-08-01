import json
import logging


class JsonFormatter(logging.Formatter):
    """Custom logging Formatter formatting standard LogRecord details as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "name": record.name,
            "level": record.levelname,
            "filename": record.filename,
            "lineno": record.lineno,
            "message": record.getMessage(),
        }
        if isinstance(record.msg, dict):
            log_record.update(record.msg)
        return json.dumps(log_record)


def setup_json_logging(level: str = "INFO") -> None:
    logger = logging.getLogger("chatii")
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(JsonFormatter())
        logger.addHandler(ch)

    access_logger = logging.getLogger("chatii.access")
    access_logger.setLevel(level)
    if not access_logger.handlers:
        ch2 = logging.StreamHandler()
        ch2.setFormatter(JsonFormatter())
        access_logger.addHandler(ch2)
