from dataclasses import dataclass
from typing import List


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    chunk_index: int
    content: str
    token_count: int


class DocumentChunker:
    """Splits raw text into overlapping chunks for semantic vector indexing."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self, doc_id: str, doc_name: str, text: str
    ) -> List[DocumentChunk]:
        words = text.split()
        if not words:
            return []

        chunks: List[DocumentChunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunk_idx = 0

        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}-chk-{chunk_idx}",
                doc_id=doc_id,
                doc_name=doc_name,
                chunk_index=chunk_idx,
                content=chunk_text,
                token_count=len(chunk_words),
            )
            chunks.append(chunk)
            chunk_idx += 1

        return chunks
