import math


class PerplexityEvaluator:
    """Computes language model perplexity from cross-entropy loss."""

    @staticmethod
    def compute(loss: float) -> float:
        """
        Calculates perplexity (exp(loss)) with clipping to prevent float overflow.
        """
        try:
            # Clip loss to avoid mathematical overflow during exponentiation
            return math.exp(min(20.0, loss))
        except OverflowError:
            return float("inf")
