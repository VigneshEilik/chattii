import re
import unicodedata


class UnicodeNormalizer:
    """
    Handles Unicode normalization, lowercasing, and whitespace standardizations.
    """

    def __init__(
        self, unicode_normalization: str = "NFKC", lowercase: bool = False
    ) -> None:
        self.unicode_normalization = unicode_normalization
        self.lowercase = lowercase
        # Matches one or more spaces and tabs
        self.space_pattern = re.compile(r"[ \t]+")
        # Matches multiple consecutive newlines
        self.newline_pattern = re.compile(r"\n+")

    def normalize(self, text: str) -> str:
        """
        Applies Unicode normalization, case transformation, and whitespace cleaning.
        """
        if not text:
            return ""

        # 1. Unicode Normalization (NFC, NFD, NFKC, NFKD, or none)
        if self.unicode_normalization and self.unicode_normalization.upper() != "NONE":
            text = unicodedata.normalize(self.unicode_normalization.upper(), text)

        # 2. Lowercase transformation
        if self.lowercase:
            text = text.lower()

        # 3. Whitespace Normalization
        # Replace multiple consecutive spaces and tabs with a single space
        text = self.space_pattern.sub(" ", text)
        # Collapse multiple newlines into a single newline
        text = self.newline_pattern.sub("\n", text)
        # Remove leading/trailing spaces
        text = text.strip()

        return text
