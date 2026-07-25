from typing import Dict, Optional, Tuple


class BPEMergeSelector:
    """
    Selects the highest frequency candidate pair with deterministic tie-breaking.
    """

    @staticmethod
    def select_best_pair(
        pair_counts: Dict[Tuple[str, str], int],
    ) -> Optional[Tuple[str, str]]:
        """
        Finds the best pair. Ties are resolved lexicographically.
        """
        if not pair_counts:
            return None

        # Min of (-count, first_token, second_token) selects highest count,
        # and breaks ties alphabetically.
        return min(pair_counts.keys(), key=lambda p: (-pair_counts[p], p[0], p[1]))
