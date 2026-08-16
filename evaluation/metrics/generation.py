import math


class GenerationMetrics:
    """Calculates generation quality metrics."""

    @staticmethod
    def exact_match(prediction: str, target: str) -> float:
        return 1.0 if prediction.strip() == target.strip() else 0.0

    @staticmethod
    def token_overlap(prediction: str, target: str) -> float:
        pred_words = set(prediction.lower().split())
        target_words = set(target.lower().split())
        if not target_words:
            return 1.0 if not pred_words else 0.0
        intersection = pred_words.intersection(target_words)
        return len(intersection) / len(target_words)

    @staticmethod
    def bleu_score(prediction: str, target: str) -> float:
        # Simplified n-gram overlap BLEU proxy
        return GenerationMetrics.token_overlap(prediction, target)

    @staticmethod
    def rouge_l(prediction: str, target: str) -> float:
        # Longest common subsequence proxy
        return GenerationMetrics.token_overlap(prediction, target)

    @staticmethod
    def perplexity(loss: float) -> float:
        return math.exp(loss)

    @staticmethod
    def context_adherence(prediction: str, context: str) -> float:
        context_words = set(context.lower().split())
        pred_words = set(prediction.lower().split())
        if not pred_words:
            return 0.0
        overlap = pred_words.intersection(context_words)
        return len(overlap) / len(pred_words)
