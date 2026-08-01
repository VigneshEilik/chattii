import logging


class ChatiiLogFormatter(logging.Formatter):
    """Formats log messages for production readability."""

    def format(self, record: logging.LogRecord) -> str:
        return f"[{record.levelname}] {record.name}: {record.getMessage()}"
