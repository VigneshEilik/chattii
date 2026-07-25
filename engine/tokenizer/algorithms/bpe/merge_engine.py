from typing import Dict, Tuple


class BPEMergeEngine:
    """
    Applies BPE merge rules on sequence arrays and corpus frequency registries.
    """

    @staticmethod
    def merge_sequence(
        sequence: Tuple[str, ...], pair: Tuple[str, str]
    ) -> Tuple[str, ...]:
        """
        Scans a sequence of tokens and replaces occurrences of (pair[0], pair[1])
        with the combined token string pair[0] + pair[1].
        """
        merged_list = []
        i = 0
        while i < len(sequence):
            if (
                i < len(sequence) - 1
                and sequence[i] == pair[0]
                and sequence[i + 1] == pair[1]
            ):
                merged_list.append(pair[0] + pair[1])
                i += 2
            else:
                merged_list.append(sequence[i])
                i += 1
        return tuple(merged_list)

    @staticmethod
    def merge_corpus(
        word_freqs: Dict[Tuple[str, ...], int], pair: Tuple[str, str]
    ) -> Dict[Tuple[str, ...], int]:
        """
        Updates the corpus counts registry by applying the merge pair
        across all sequences.
        """
        new_word_freqs: Dict[Tuple[str, ...], int] = {}
        for seq, count in word_freqs.items():
            new_seq = BPEMergeEngine.merge_sequence(seq, pair)
            new_word_freqs[new_seq] = new_word_freqs.get(new_seq, 0) + count
        return new_word_freqs
