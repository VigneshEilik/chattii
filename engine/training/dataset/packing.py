# ruff: noqa: E501
from typing import Generator, List


class SequencePacker:
    """Packs documents' variable-length tokens list into fixed-size sequences."""

    @staticmethod
    def pack(
        token_lists: List[List[int]], sequence_length: int, eos_token_id: int = 0
    ) -> Generator[List[int], None, None]:
        """
        Packs lists of tokens together, inserting an EOS boundary token between them,
        yielding sequences of size exactly (sequence_length + 1) for next-token shifting.
        """
        current_block: List[int] = []
        target_size = sequence_length + 1

        for tokens in token_lists:
            # Concatenate document tokens followed by the separator EOS boundary token
            tokens_to_pack = tokens + [eos_token_id]
            for t in tokens_to_pack:
                current_block.append(t)
                if len(current_block) == target_size:
                    yield current_block
                    current_block = []

        # If a partial block remains, pad with EOS token to reach full sequence block size
        if current_block:
            padding_len = target_size - len(current_block)
            current_block.extend([eos_token_id] * padding_len)
            yield current_block
