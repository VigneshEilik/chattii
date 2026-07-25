from typing import Tuple

from engine.dataset.base.dataset import TextSample
from engine.dataset.base.validator import BaseValidator


class EncodingValidator(BaseValidator):
    """Ensures text can be cleanly encoded to UTF-8 without serialization issues."""

    def validate(self, sample: TextSample) -> Tuple[bool, str]:
        try:
            sample.text.encode("utf-8")
            return True, ""
        except UnicodeEncodeError as e:
            return False, f"Unicode encoding error: {e}"
