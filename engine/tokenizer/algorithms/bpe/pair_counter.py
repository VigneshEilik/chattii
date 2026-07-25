from typing import Dict, Tuple


class BPETokenPairCounter:
    """
    Counts frequencies of adjacent subword token pairs inside corpus sequences.
    """

    @staticmethod
    def count_pairs(
        word_freqs: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, str], int]:
        """
        Scans sequence dictionaries and tallies adjacent subword pairs.
        """
        pair_counts: Dict[Tuple[str, str], int] = {}
        for sequence, count in word_freqs.items():
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        return pair_counts
