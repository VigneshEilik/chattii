import unicodedata

from engine.dataset.base.cleaner import BaseCleaner
from engine.dataset.base.dataset import TextSample


class UnicodeCleaner(BaseCleaner):
    """Applies canonical Unicode normalization (NFC or NFKC)."""

    def __init__(self, form: str = "NFC") -> None:
        self.form = form

    def clean(self, sample: TextSample) -> TextSample:
        normalized = unicodedata.normalize(self.form, sample.text)
        return TextSample(
            text=normalized,
            sample_id=sample.sample_id,
            metadata=sample.metadata,
        )
