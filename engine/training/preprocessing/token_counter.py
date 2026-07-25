from typing import Optional

from engine.tokenizer.service import TokenizerService


class TokenCounter:
    """Estimates or exactly counts token length of raw text segments."""

    @staticmethod
    def count(text: str, tokenizer: Optional[TokenizerService] = None) -> int:
        """
        Calculates exact tokens count if a tokenizer is provided,
        else estimates count using word multiplier heuristics (~1.3x words).
        """
        if not text:
            return 0

        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text))
            except Exception:
                pass

        # Heuristic fallback: ~1.3 tokens per whitespace word
        return max(1, int(len(text.split()) * 1.3))
