from engine.dataset.base.dataset import TextSample
from engine.dataset.base.loader import BaseLoader
from engine.dataset.cleaner.whitespace import WhitespaceCleaner
from engine.dataset.validator.quality import QualityValidator
from engine.dataset.loader.streaming import StreamingPipeline
from typing import Iterator


class MockLoader(BaseLoader):
    def stream(self) -> Iterator[TextSample]:
        yield TextSample("   Clean sample 1  ")
        yield TextSample("Short")
        yield TextSample("   Clean sample 2  ")


def test_streaming_pipeline() -> None:
    loader = MockLoader()
    cleaners = [WhitespaceCleaner()]
    validators = [QualityValidator(min_length=10)]

    pipeline = StreamingPipeline(loader, cleaners=cleaners, validators=validators)
    samples = list(pipeline.stream())

    # "Short" gets filtered because length < 10.
    # The others get cleaned.
    assert len(samples) == 2
    assert samples[0].text == "Clean sample 1"
    assert samples[1].text == "Clean sample 2"
