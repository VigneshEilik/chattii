import re

from engine.dataset.base.cleaner import BaseCleaner
from engine.dataset.base.dataset import TextSample


class WhitespaceCleaner(BaseCleaner):
    """Trims leading/trailing space and collapses duplicate spaces and newlines."""

    def clean(self, sample: TextSample) -> TextSample:
        # Split by newlines, trim intermediate spaces, drop blank lines, and join
        lines = [
            re.sub(r"[ \t]+", " ", line.strip()) for line in sample.text.split("\n")
        ]
        text = "\n".join(line for line in lines if line)
        return TextSample(
            text=text.strip(),
            sample_id=sample.sample_id,
            metadata=sample.metadata,
        )
