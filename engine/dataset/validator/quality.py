import re
from typing import Tuple

from engine.dataset.base.dataset import TextSample
from engine.dataset.base.validator import BaseValidator


class QualityValidator(BaseValidator):
    """Filters low-quality samples using length thresholds and special character/word ratios."""

    def __init__(self, min_length: int = 5, max_length: int = 100000) -> None:
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, sample: TextSample) -> Tuple[bool, str]:
        text = sample.text
        char_len = len(text)

        if char_len < self.min_length:
            return False, f"Length ({char_len}) is below minimum threshold ({self.min_length})"
        if char_len > self.max_length:
            return False, f"Length ({char_len}) exceeds maximum threshold ({self.max_length})"

        # Heuristic 1: Special character ratio
        special_chars = len(re.findall(r"[^a-zA-Z0-9\s]", text))
        ratio = special_chars / char_len
        if ratio > 0.45:
            return False, f"Excessive special symbol ratio: {ratio:.2%}"

        # Heuristic 2: Average word length check
        words = text.split()
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len < 2.0 or avg_word_len > 15.0:
                return False, f"Out-of-bound average word length: {avg_word_len:.2f}"

        return True, ""
