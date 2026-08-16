from engine.rag.chunker import DocumentChunker
from engine.rag.parsers import DocumentParser
from engine.rag.vector_store import VectorStore, cosine_similarity, simple_embedding


def test_parse_text():
    assert DocumentParser.parse_text("  hello world  ") == "hello world"


def test_parse_json():
    res = DocumentParser.parse_json('{"key": "value"}')
    assert "key" in res and "value" in res


def test_parse_csv():
    csv_raw = "a,b\n1,2"
    res = DocumentParser.parse_csv(csv_raw)
    assert "a, b" in res


def test_parse_file_fallback():
    res = DocumentParser.parse_file("doc.txt", b"plain text")
    assert res == "plain text"


def test_chunk_document():
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
    text = " ".join([f"word{i}" for i in range(25)])
    chunks = chunker.chunk_document("doc-1", "doc.txt", text)
    assert len(chunks) > 0
    assert chunks[0].doc_id == "doc-1"


def test_chunk_empty_document():
    chunker = DocumentChunker()
    assert chunker.chunk_document("doc-2", "doc.txt", "") == []


def test_simple_embedding():
    vec = simple_embedding("test query", dim=64)
    assert len(vec) == 64


def test_cosine_similarity_identical():
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == 1.0


def test_vector_store_add_and_search():
    store = VectorStore(dim=64)
    chunker = DocumentChunker(chunk_size=5, chunk_overlap=0)
    chunks = chunker.chunk_document(
        "doc-3", "ml.txt", "gradient descent optimization in machine learning"
    )
    for c in chunks:
        store.add_chunk(c)

    results = store.search("gradient descent", top_k=1)
    assert len(results) == 1
    assert "gradient" in results[0][0].content.lower()


def test_vector_store_delete():
    store = VectorStore(dim=64)
    chunker = DocumentChunker(chunk_size=5, chunk_overlap=0)
    chunks = chunker.chunk_document("doc-4", "del.txt", "text content")
    for c in chunks:
        store.add_chunk(c)

    deleted_count = store.delete_document_chunks("doc-4")
    assert deleted_count == len(chunks)
    assert store.search("text", top_k=5) == []


def test_vector_store_min_score():
    store = VectorStore(dim=64)
    chunker = DocumentChunker()
    chunks = chunker.chunk_document("doc-5", "f.txt", "python coding")
    for c in chunks:
        store.add_chunk(c)
    assert (
        store.search("completely unrelated query zzz", top_k=1, min_score=0.999) == []
    )


def test_chunk_index_ordering():
    chunker = DocumentChunker(chunk_size=5, chunk_overlap=0)
    chunks = chunker.chunk_document("doc-6", "f.txt", "a b c d e f g h i j")
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
