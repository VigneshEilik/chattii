from collections import Counter
from typing import Any, Dict, List


class DatasetStatistics:
    """Computes distribution metrics and statistics over token sequences."""

    @staticmethod
    def compute(token_blocks: List[List[int]]) -> Dict[str, Any]:
        """
        Gathers count stats, token diversity, and frequency statistics.
        """
        if not token_blocks:
            return {
                "num_sequences": 0,
                "total_tokens": 0,
                "vocab_diversity": 0,
                "most_common_tokens": [],
            }

        total_tokens = sum(len(block) for block in token_blocks)
        counter = Counter(t for block in token_blocks for t in block)

        return {
            "num_sequences": len(token_blocks),
            "total_tokens": total_tokens,
            "vocab_diversity": len(counter),
            "most_common_tokens": counter.most_common(5),
        }
