from typing import Iterator

from engine.dataset.base.dataset import TokenizedSample
from engine.dataset.base.packer import BasePacker


class SequencePacker(BasePacker):
    """Packs variable-length token sequences into fixed-length context windows with EOS boundaries."""

    def __init__(
        self,
        max_sequence_length: int = 2048,
        add_eos: bool = True,
        eos_token_id: int = 2,
    ) -> None:
        self.max_sequence_length = max_sequence_length
        self.add_eos = add_eos
        self.eos_token_id = eos_token_id

    def pack(self, samples: Iterator[TokenizedSample]) -> Iterator[TokenizedSample]:
        buffer = []
        packed_idx = 0

        for sample in samples:
            ids = sample.input_ids
            if self.add_eos:
                ids = ids + [self.eos_token_id]

            buffer.extend(ids)
            while len(buffer) >= self.max_sequence_length:
                chunk = buffer[: self.max_sequence_length]
                buffer = buffer[self.max_sequence_length :]

                yield TokenizedSample(
                    input_ids=chunk,
                    sample_id=f"packed_{packed_idx}",
                    attention_mask=[1] * self.max_sequence_length,
                    labels=chunk.copy(),
                )
                packed_idx += 1
