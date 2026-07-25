from typing import Iterator

from engine.dataset.base.dataset import TextSample, TokenizedSample
from engine.dataset.tokenizer.adapter import TokenizerAdapter


class TokenizationPipeline:
    """Consumes a stream of TextSamples and yields TokenizedSamples."""

    def __init__(self, adapter: TokenizerAdapter) -> None:
        self.adapter = adapter

    def tokenize(self, samples: Iterator[TextSample]) -> Iterator[TokenizedSample]:
        for sample in samples:
            yield self.adapter.process(sample)
