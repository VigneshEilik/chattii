from typing import Dict, List, Optional, Tuple


class TokenizerCache:
    """In-memory cache mapping input strings to token IDs (and vice versa)."""

    def __init__(self) -> None:
        self._encode_cache: Dict[str, List[int]] = {}
        self._decode_cache: Dict[Tuple[int, ...], str] = {}

    def get_ids(self, text: str) -> Optional[List[int]]:
        return self._encode_cache.get(text)

    def set_ids(self, text: str, ids: List[int]) -> None:
        self._encode_cache[text] = ids

    def get_text(self, ids: List[int]) -> Optional[str]:
        return self._decode_cache.get(tuple(ids))

    def set_text(self, ids: List[int], text: str) -> None:
        self._decode_cache[tuple(ids)] = text

    def clear(self) -> None:
        self._encode_cache.clear()
        self._decode_cache.clear()
