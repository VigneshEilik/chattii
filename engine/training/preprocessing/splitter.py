from typing import List


class DocumentSplitter:
    """Splits long documents into overlapping, size-bounded text chunks."""

    @staticmethod
    def split(text: str, max_words: int = 500, overlap: int = 50) -> List[str]:
        """
        Splits text on word boundaries using sliding windows with optional overlap.
        """
        if not text:
            return []

        words = text.split()
        if len(words) <= max_words:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            if end == len(words):
                break
            # Advance starting index by window step
            start += max_words - overlap
            if start >= end:
                start = end - 1

        return chunks
