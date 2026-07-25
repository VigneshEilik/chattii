from typing import Iterator, List

from engine.dataset.base.dataset import TokenizedSample
from engine.dataset.packing.packer import SequencePacker


class DynamicPacker(SequencePacker):
    """Packs sequences dynamically up to a specified maximum length, yielding early if needed."""

    def pack(self, samples: Iterator[TokenizedSample]) -> Iterator[TokenizedSample]:
        current_seq: List[int] = []
        packed_idx = 0

        for sample in samples:
            ids = sample.input_ids
            if self.add_eos:
                ids = ids + [self.eos_token_id]

            if len(current_seq) + len(ids) > self.max_sequence_length:
                if current_seq:
                    # Yield current accumulated sequence
                    yield TokenizedSample(
                        input_ids=current_seq,
                        sample_id=f"dyn_packed_{packed_idx}",
                        attention_mask=[1] * len(current_seq),
                        labels=current_seq.copy(),
                    )
                    packed_idx += 1
                current_seq = ids
            else:
                current_seq.extend(ids)

        if current_seq:
            yield TokenizedSample(
                input_ids=current_seq,
                sample_id=f"dyn_packed_{packed_idx}",
                attention_mask=[1] * len(current_seq),
                labels=current_seq.copy(),
            )
