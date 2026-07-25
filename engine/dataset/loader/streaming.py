from typing import Iterator, List, Optional

from engine.dataset.base.cleaner import BaseCleaner
from engine.dataset.base.dataset import TextSample
from engine.dataset.base.loader import BaseLoader
from engine.dataset.base.validator import BaseValidator


class StreamingPipeline:
    """Chains a data loader, multiple text cleaners, and quality validators into a memory-efficient generator."""

    def __init__(
        self,
        loader: BaseLoader,
        cleaners: Optional[List[BaseCleaner]] = None,
        validators: Optional[List[BaseValidator]] = None,
    ) -> None:
        self.loader = loader
        self.cleaners = cleaners or []
        self.validators = validators or []

    def stream(self) -> Iterator[TextSample]:
        for sample in self.loader:
            # Apply all text cleaning filters sequentially
            cleaned = sample
            for cleaner in self.cleaners:
                cleaned = cleaner.clean(cleaned)

            # Apply all quality validations
            is_valid = True
            for validator in self.validators:
                valid, _ = validator.validate(cleaned)
                if not valid:
                    is_valid = False
                    break

            if is_valid:
                yield cleaned

    def __iter__(self) -> Iterator[TextSample]:
        return self.stream()
