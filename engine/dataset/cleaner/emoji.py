import re

from engine.dataset.base.cleaner import BaseCleaner
from engine.dataset.base.dataset import TextSample


class EmojiCleaner(BaseCleaner):
    """Filters or removes emoji symbols from text."""

    def __init__(self, replace_with: str = "") -> None:
        self.replace_with = replace_with
        # Matches typical Unicode ranges dedicated to emojis/pictographs
        self.emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f1e0-\U0001f1ff"
            "\U00002700-\U000027bf"
            "\U00002600-\U000026ff"
            "]+",
            flags=re.UNICODE,
        )

    def clean(self, sample: TextSample) -> TextSample:
        text = self.emoji_pattern.sub(self.replace_with, sample.text)
        return TextSample(
            text=text,
            sample_id=sample.sample_id,
            metadata=sample.metadata,
        )
