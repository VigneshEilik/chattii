import io
import json
import logging

from core.utils.logging import JSONFormatter


def test_json_formatter_metadata_parsing() -> None:
    """
    Verifies that the JSONFormatter converts records to structured JSON
    and captures additional variables passed through the `extra` dictionary.
    """
    logger = logging.getLogger("test_json_logger")
    logger.setLevel(logging.INFO)

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    try:
        # Log statement with extra structured metrics
        logger.info(
            "Training step metrics logged",
            extra={"epoch": 3, "step": 4500, "loss": 1.42},
        )

        # Parse output JSON
        log_output = log_capture.getvalue().strip()
        parsed_log = json.loads(log_output)

        # Assert structure
        assert "timestamp" in parsed_log
        assert parsed_log["level"] == "INFO"
        assert parsed_log["module"] == "test_json_logger"
        assert parsed_log["message"] == "Training step metrics logged"
        assert parsed_log["epoch"] == 3
        assert parsed_log["step"] == 4500
        assert parsed_log["loss"] == 1.42

    finally:
        # Clean up handler
        logger.removeHandler(handler)
        handler.close()
