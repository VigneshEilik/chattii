from typing import Any, Dict, List, Tuple


class BPEMergeHistory:
    """
    Maintains a list of all applied BPE merges for audit and statistics.
    """

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def record(
        self, rank: int, pair: Tuple[str, str], merged: str, frequency: int
    ) -> None:
        """
        Records a single merge event.
        """
        self.history.append(
            {
                "rank": rank,
                "pair": [pair[0], pair[1]],
                "merged": merged,
                "frequency": frequency,
            }
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Returns the full merge history list.
        """
        return self.history
