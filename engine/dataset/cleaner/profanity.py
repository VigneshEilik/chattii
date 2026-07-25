from engine.dataset.base.cleaner import BaseCleaner
from engine.dataset.base.dataset import TextSample


class ProfanityCleaner(BaseCleaner):
    """Filters or redacts toxic words / profanities (keyword-based stub)."""

    def __init__(self, redact: bool = True) -> None:
        self.redact = redact
        # Simple sample list; in production, this is loaded from a dictionary file
        self.banned_keywords = {"bannedword1", "bannedword2"}

    def clean(self, sample: TextSample) -> TextSample:
        text = sample.text
        metadata = sample.metadata.copy()

        found = []
        for word in self.banned_keywords:
            if word in text.lower():
                found.append(word)
                if self.redact:
                    # Case insensitive replace
                    import re
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    text = pattern.sub(" [REDACTED] ", text)

        if found:
            metadata["contains_profanity"] = True
            metadata["flagged_words"] = found

        return TextSample(
            text=text,
            sample_id=sample.sample_id,
            metadata=metadata,
        )
