from engine.dataset.base.dataset import TextSample


class BaseCleaner:
    """Abstract interface for text normalization and cleaning filters."""

    def clean(self, sample: TextSample) -> TextSample:
        raise NotImplementedError
