from typing import Any, Dict


class BPEStatistics:
    """
    Computes quality and compression statistics during and after BPE training.
    """

    @staticmethod
    def compute(
        vocab_size: int,
        num_merges: int,
        initial_corpus_bytes: int,
        current_corpus_tokens_count: int,
        vocab_map: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Calculates BPE specific metrics.
        """
        # Exclude common special tokens from average length calculations
        non_specials = [
            t for t in vocab_map.keys() if not (t.startswith("<") and t.endswith(">"))
        ]

        total_len = sum(len(t) for t in non_specials)
        avg_len = total_len / len(non_specials) if non_specials else 0.0

        # Compression ratio: (Raw bytes size) / (Current tokenized tokens)
        compression_ratio = (
            initial_corpus_bytes / current_corpus_tokens_count
            if current_corpus_tokens_count > 0
            else 1.0
        )

        return {
            "total_merges": num_merges,
            "vocab_size": vocab_size,
            "average_token_length": round(avg_len, 4),
            "compression_ratio": round(compression_ratio, 4),
            "current_tokens_count": current_corpus_tokens_count,
        }
