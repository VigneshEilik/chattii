from typing import List

from engine.dataset.base.dataset import TextSample, TokenizedSample
from engine.tokenizer.service import TokenizerService


class TokenizerAdapter:
    """Bridges streaming text samples with Chatii's TokenizerService."""

    def __init__(self, algorithm: str = "bpe", vocab_dir: str = "models/tokenizer") -> None:
        self.tokenizer = TokenizerService(algorithm)
        self.tokenizer.load(vocab_dir)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def process(self, sample: TextSample) -> TokenizedSample:
        ids = self.encode(sample.text)
        return TokenizedSample(
            input_ids=ids,
            sample_id=sample.sample_id,
            metadata=sample.metadata.copy()
        )
