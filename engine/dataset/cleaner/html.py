import html
import re

from engine.dataset.base.cleaner import BaseCleaner
from engine.dataset.base.dataset import TextSample


class HTMLCleaner(BaseCleaner):
    """Strips HTML tags and unescapes common HTML character entities."""

    def clean(self, sample: TextSample) -> TextSample:
        # Strip all HTML-like tags using regex
        text = re.sub(r"<[^>]*>", "", sample.text)
        # Unescape HTML character entities (e.g., &amp; -> &)
        text = html.unescape(text)
        return TextSample(
            text=text,
            sample_id=sample.sample_id,
            metadata=sample.metadata,
        )
