import hashlib
from typing import List, Set, Tuple

from engine.dataset.base.dataset import TextSample
from engine.dataset.base.validator import BaseValidator


class DuplicateValidator(BaseValidator):
    """Detects exact duplicates via SHA-256 and near-duplicates using a MinHash signature skeleton."""

    def __init__(self, exact: bool = True, near_dup: bool = True, threshold: float = 0.85) -> None:
        self.exact = exact
        self.near_dup = near_dup
        self.threshold = threshold
        self.seen_hashes: Set[str] = set()
        self.seen_signatures: List[List[int]] = []

    def _get_shingles(self, text: str, k: int = 3) -> Set[Tuple[str, ...]]:
        words = text.lower().split()
        if len(words) < k:
            return {tuple(words)}
        return {tuple(words[i:i+k]) for i in range(len(words) - k + 1)}

    def _minhash_signature(self, shingles: Set[Tuple[str, ...]], num_hashes: int = 16) -> List[int]:
        sig = []
        for i in range(num_hashes):
            min_val = 0xFFFFFFFFFFFFFFFF
            for shingle in shingles:
                shingle_str = "".join(shingle)
                # Compute md5 shingle hash with a unique seed
                h_val = int(
                    hashlib.md5(f"{shingle_str}_{i}".encode("utf-8")).hexdigest(),
                    16
                )
                if h_val < min_val:
                    min_val = h_val
            sig.append(min_val)
        return sig

    def validate(self, sample: TextSample) -> Tuple[bool, str]:
        cleaned = sample.text.strip().lower()
        if not cleaned:
            return False, "Empty text"

        # 1. Exact Deduplication
        if self.exact:
            h = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            if h in self.seen_hashes:
                return False, "Exact duplicate found"
            self.seen_hashes.add(h)

        # 2. Near-Duplicate (MinHash) Detection
        if self.near_dup:
            shingles = self._get_shingles(cleaned)
            sig = self._minhash_signature(shingles)

            for existing_sig in self.seen_signatures:
                matches = sum(1 for a, b in zip(sig, existing_sig) if a == b)
                similarity = matches / len(sig)
                if similarity >= self.threshold:
                    return False, f"Near duplicate found (Jaccard similarity: {similarity:.2%})"
            self.seen_signatures.append(sig)

        return True, ""
