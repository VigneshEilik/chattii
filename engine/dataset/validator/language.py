from typing import Tuple

from engine.dataset.base.dataset import TextSample
from engine.dataset.base.validator import BaseValidator


class LanguageValidator(BaseValidator):
    """Filters samples that do not match the target language (with heuristic fallback)."""

    def __init__(self, target_lang: str = "en") -> None:
        self.target_lang = target_lang

    def validate(self, sample: TextSample) -> Tuple[bool, str]:
        if not sample.text.strip():
            return False, "Empty text"

        try:
            # Try to use langdetect if installed
            from langdetect import detect

            lang = detect(sample.text)
            if lang == self.target_lang:
                return True, ""
            return False, f"Language mismatch: expected {self.target_lang}, got {lang}"
        except ImportError:
            # Fallback heuristic for English
            if self.target_lang == "en":
                non_ascii = sum(1 for c in sample.text if ord(c) > 127)
                ratio = non_ascii / max(1, len(sample.text))
                if ratio > 0.35:
                    return (
                        False,
                        f"High non-ASCII ratio ({ratio:.2%}) for English target",
                    )
            return True, ""
