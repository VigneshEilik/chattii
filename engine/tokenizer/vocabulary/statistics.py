import math
from typing import Any

from pydantic import BaseModel


class VocabularyStatistics(BaseModel):
    """
    Evaluation metrics representing vocabulary qualities and distribution states.
    """

    vocab_size: int
    reserved_tokens_count: int
    unique_tokens_count: int
    average_token_length: float
    longest_token: str
    shortest_token: str
    longest_token_length: int
    shortest_token_length: int
    token_entropy: float

    @classmethod
    def compute(cls, manager: Any) -> "VocabularyStatistics":
        """
        Calculates all metric values from a VocabularyManager instance.
        """
        specials = set(manager.get_special_tokens().values())
        non_specials = [t for t in manager._token_to_id.keys() if t not in specials]

        vocab_size = len(manager)
        reserved_count = len(specials)
        unique_count = len(non_specials)

        if not non_specials:
            return cls(
                vocab_size=vocab_size,
                reserved_tokens_count=reserved_count,
                unique_tokens_count=0,
                average_token_length=0.0,
                longest_token="",
                shortest_token="",
                longest_token_length=0,
                shortest_token_length=0,
                token_entropy=0.0,
            )

        total_len = sum(len(t) for t in non_specials)
        avg_len = total_len / len(non_specials)

        longest = max(non_specials, key=len)
        shortest = min(non_specials, key=len)

        # Entropy calculation based on observed training token frequencies
        total_freq = sum(manager._token_frequencies.values())
        entropy = 0.0
        if total_freq > 0:
            for token in non_specials:
                freq = manager._token_frequencies.get(token, 0)
                if freq > 0:
                    prob = freq / total_freq
                    entropy -= prob * math.log2(prob)

        return cls(
            vocab_size=vocab_size,
            reserved_tokens_count=reserved_count,
            unique_tokens_count=unique_count,
            average_token_length=round(avg_len, 4),
            longest_token=longest,
            shortest_token=shortest,
            longest_token_length=len(longest),
            shortest_token_length=len(shortest),
            token_entropy=round(entropy, 4),
        )
