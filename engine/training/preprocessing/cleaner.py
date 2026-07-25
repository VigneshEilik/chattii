import unicodedata


class TextCleaner:
    """Cleans and standardizes raw text inputs."""

    @staticmethod
    def clean(text: str) -> str:
        """
        Strips non-printable/control characters, standardizes whitespaces,
        and normalizes Unicode representation to NFC.
        """
        if not text:
            return ""

        # Normalize unicode representation
        text = unicodedata.normalize("NFC", text)

        # Filter out control characters except newline and tab
        cleaned_chars = [
            char
            for char in text
            if not unicodedata.category(char).startswith("C") or char in ["\n", "\t"]
        ]
        text = "".join(cleaned_chars)

        # Standardize multiple consecutive spaces to a single space, preserving newlines
        lines = text.split("\n")
        processed_lines = []
        for line in lines:
            words = line.split()
            processed_lines.append(" ".join(words))

        return "\n".join(processed_lines).strip()
