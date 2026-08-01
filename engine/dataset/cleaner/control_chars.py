import unicodedata

from engine.dataset.base.cleaner import BaseCleaner
from engine.dataset.base.dataset import TextSample


class ControlCharacterCleaner(BaseCleaner):
    """Strips Unicode control characters while retaining standard whitespace formatting."""

    def clean(self, sample: TextSample) -> TextSample:
        # Preserve newlines and tabs, strip other control characters ('Cc' category)
        cleaned = "".join(
            c
            for c in sample.text
            if unicodedata.category(c) != "Cc" or c in ("\n", "\t")
        )
        return TextSample(
            text=cleaned,
            sample_id=sample.sample_id,
            metadata=sample.metadata,
        )
