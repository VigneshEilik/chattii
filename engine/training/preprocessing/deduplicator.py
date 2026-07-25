import hashlib
from typing import Set


class DocumentDeduplicator:
    """Removes duplicate documents using hash fingerprint tracking."""

    def __init__(self) -> None:
        self.seen_hashes: Set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        """
        Computes normalized content hashes and checks for duplicates.
        """
        normalized = "".join(text.split()).lower()
        if not normalized:
            return True

        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if content_hash in self.seen_hashes:
            return True

        self.seen_hashes.add(content_hash)
        return False

    def clear(self) -> None:
        """Resets seen hash database."""
        self.seen_hashes.clear()
