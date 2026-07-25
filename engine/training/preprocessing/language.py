from typing import Set


class LanguageDetector:
    """Heuristic language filtering using common stopword occurrences."""

    ENGLISH_STOPWORDS: Set[str] = {
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
    }

    @staticmethod
    def is_target_language(
        text: str, target: str = "en", threshold: float = 0.05
    ) -> bool:
        """
        Determines if text is in the target language (default 'en')
        by validating the frequency of common language stopwords.
        """
        if not text:
            return False

        if target != "en":
            # Return True for other languages when stopwords are undefined
            return True

        words = [w.lower().strip(",.?!:;\"()[]{}'") for w in text.split()]
        if not words:
            return False

        match_count = sum(1 for w in words if w in LanguageDetector.ENGLISH_STOPWORDS)
        ratio = match_count / len(words)
        return ratio >= threshold
