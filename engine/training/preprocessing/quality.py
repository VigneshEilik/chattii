class QualityFilter:
    """Applies heuristic rules to filter out low-quality documents."""

    @staticmethod
    def is_high_quality(
        text: str,
        min_words: int = 5,
        max_repetitive_fraction: float = 0.5,
        max_non_alpha_ratio: float = 0.4,
    ) -> bool:
        """
        Validates text size, repetitive patterns, and alphabet character density.
        """
        if not text:
            return False

        words = text.split()
        if len(words) < min_words:
            return False

        # 1. Repetitive patterns check
        unique_words = set(words)
        repeat_ratio = 1.0 - (len(unique_words) / len(words))
        if repeat_ratio > max_repetitive_fraction:
            return False

        # 2. Non-alphanumeric ratio check
        clean_chars = sum(1 for c in text if c.isalnum() or c.isspace())
        non_alpha_ratio = 1.0 - (clean_chars / len(text))
        if non_alpha_ratio > max_non_alpha_ratio:
            return False

        return True
