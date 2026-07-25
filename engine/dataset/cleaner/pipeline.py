from typing import List, Optional

from engine.dataset.base.cleaner import BaseCleaner
from engine.dataset.base.dataset import TextSample


class CleaningPipeline(BaseCleaner):
    """Executes a list of cleaning filters sequentially on text samples."""

    def __init__(self, cleaners: Optional[List[BaseCleaner]] = None) -> None:
        self.cleaners = cleaners or []

    def clean(self, sample: TextSample) -> TextSample:
        cleaned = sample
        for cleaner in self.cleaners:
            cleaned = cleaner.clean(cleaned)
        return cleaned
