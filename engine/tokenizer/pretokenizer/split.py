import re
from typing import List, Optional


class RegexPretokenizer:
    """
    Splits normalized text into base structural string components
    (pre-tokens) before training or encoding with BPE.
    """

    def __init__(
        self, mode: str = "word", custom_pattern: Optional[str] = None
    ) -> None:
        self.mode = mode

        if custom_pattern:
            self.pattern = re.compile(custom_pattern)
        elif mode == "word":
            # Matches English contractions, words/digits (with optional leading space),
            # punctuation groups, and consecutive space patterns.
            self.pattern = re.compile(
                r"'s|'t|'re|'ve|'m|'ll|'d|"
                r" ?\w+|"
                r" ?[^\s\w]+|"
                r"\s+(?!\S)|\s+"
            )
        else:
            self.pattern = None

    def pretokenize(self, text: str) -> List[str]:
        """
        Splits text into a list of pre-tokens.
        """
        if not text:
            return []

        if self.mode == "char":
            return list(text)

        if self.pattern:
            # Match regex segments
            matches = self.pattern.findall(text)
            return [m for m in matches if m]

        # Simple whitespace fallback
        return text.split(" ")
