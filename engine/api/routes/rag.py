import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from engine.rag.chunker import DocumentChunker
from engine.rag.parsers import DocumentParser
from engine.rag.vector_store import get_vector_store

router = APIRouter(prefix="/v1/rag", tags=["RAG Knowledge"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    min_score: float = 0.0


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):  # noqa: B008
    try:
        content_bytes = await file.read()
        raw_text = DocumentParser.parse_file(file.filename, content_bytes)

        doc_id = f"doc-{uuid.uuid4().hex[:12]}"
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc_id, file.filename, raw_text)

        store = get_vector_store()
        for c in chunks:
            store.add_chunk(c)

        return {
            "document_id": doc_id,
            "filename": file.filename,
            "chunks_indexed": len(chunks),
            "status": "indexed",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest document: {e}")


@router.post("/search")
def search_knowledge(req: SearchRequest):
    store = get_vector_store()
    results = store.search(query=req.query, top_k=req.top_k, min_score=req.min_score)
    return [
        {
            "chunk_id": chunk.chunk_id,
            "doc_name": chunk.doc_name,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "score": round(score, 4),
        }
        for chunk, score in results
    ]
