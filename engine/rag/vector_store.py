import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from engine.rag.chunker import DocumentChunk


def simple_embedding(text: str, dim: int = 128) -> List[float]:
    """Generates a deterministic pseudo-embedding vector for text."""
    vec = [0.0] * dim
    for i, word in enumerate(text.lower().split()):
        idx = hash(word) % dim
        vec[idx] += 1.0 / (i + 1)
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot


@dataclass
class VectorRecord:
    chunk: DocumentChunk
    vector: List[float]


class VectorStore:
    """In-memory vector store with cosine similarity retrieval."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self._store: Dict[str, VectorRecord] = {}

    def add_chunk(self, chunk: DocumentChunk) -> None:
        vec = simple_embedding(chunk.content, self.dim)
        self._store[chunk.chunk_id] = VectorRecord(chunk=chunk, vector=vec)

    def search(
        self, query: str, top_k: int = 3, min_score: float = 0.0
    ) -> List[Tuple[DocumentChunk, float]]:
        query_vec = simple_embedding(query, self.dim)
        results = []

        for record in self._store.values():
            score = cosine_similarity(query_vec, record.vector)
            if score >= min_score:
                results.append((record.chunk, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete_document_chunks(self, doc_id: str) -> int:
        to_delete = [
            cid for cid, rec in self._store.items() if rec.chunk.doc_id == doc_id
        ]
        for cid in to_delete:
            self._store.pop(cid, None)
        return len(to_delete)


_vector_store_instance: VectorStore = VectorStore()


def get_vector_store() -> VectorStore:
    return _vector_store_instance
