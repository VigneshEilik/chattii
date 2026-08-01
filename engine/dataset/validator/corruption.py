from typing import Tuple

from engine.dataset.base.dataset import TextSample
from engine.dataset.base.validator import BaseValidator


class CorruptionValidator(BaseValidator):
    """Checks for data corruption, zero-byte payloads, or unreadable strings."""

    def validate(self, sample: TextSample) -> Tuple[bool, str]:
        if sample.text is None:
            return False, "Null sample payload"
        if not isinstance(sample.text, str):
            return (
                False,
                f"Invalid type: expected string, got {type(sample.text).__name__}",
            )
        if not sample.text.strip():
            return False, "Empty or white-space-only sample"
        return True, ""
