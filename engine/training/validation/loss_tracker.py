from typing import List


class LossTracker:
    """Tracks historical progression of training/validation loss metrics."""

    def __init__(self) -> None:
        self.history: List[float] = []

    def add(self, loss: float) -> None:
        """Appends a new loss metric value to history."""
        self.history.append(loss)

    def best(self) -> float:
        """Returns the lowest (best) loss recorded in history."""
        return min(self.history) if self.history else float("inf")

    def recent(self) -> float:
        """Returns the most recently recorded loss value."""
        return self.history[-1] if self.history else float("inf")
