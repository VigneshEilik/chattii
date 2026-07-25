from typing import Any, List, Tuple

from engine.tokenizer.algorithms.bpe.exceptions import BPEError


class BPEValidator:
    """
    Verifies BPE properties, checks for merge rule conflicts, and validates roundtrips.
    """

    @staticmethod
    def validate_merges(merges: List[Tuple[str, str]]) -> None:
        """
        Validates that merges has no duplicates and contains valid structures.
        """
        seen = set()
        for idx, pair in enumerate(merges):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise BPEError(f"Invalid BPE merge rule format at index {idx}: {pair}")
            if pair in seen:
                raise BPEError(f"Duplicate BPE merge rule registered: {pair}")
            seen.add(pair)

    @staticmethod
    def validate_roundtrip(tokenizer_instance: Any, text: str) -> None:
        """
        Asserts roundtrip identity of encode + decode.
        """
        # We type hint Any here to avoid circular imports with BPETokenizer
        try:
            ids = tokenizer_instance.encode(text)
            decoded = tokenizer_instance.decode(ids)
        except Exception as e:
            raise BPEError(f"Exception during BPE roundtrip processing: {e}") from e

        if text != decoded:
            raise BPEError(
                f"BPE roundtrip mismatch!\n"
                f"Input text  : '{text}'\n"
                f"Decoded text: '{decoded}'"
            )
