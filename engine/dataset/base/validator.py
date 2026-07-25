from typing import Tuple

from engine.dataset.base.dataset import TextSample


class BaseValidator:
    """Abstract interface for text quality validation filters."""

    def validate(self, sample: TextSample) -> Tuple[bool, str]:
        """
        Validates sample quality.
        Returns (is_valid, reason_if_invalid).
        """
        raise NotImplementedError
