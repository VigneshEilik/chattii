# Chatii RAG Knowledge System

The Chatii RAG system provides semantic context retrieval for grounded AI assistant conversations.

## Pipeline Architecture
1. **Document Ingestion**: Supports PDF, TXT, Markdown, DOCX, CSV, JSON, HTML.
2. **Chunking**: Recursive token & sentence-aware document chunking (`DocumentChunker`).
3. **Embeddings & Vector Store**: Cosine similarity vector search indexed in `VectorStore`.
4. **Retrieval**: Returns relevance scored chunks with source document citations.

## API Endpoint
`POST /v1/rag/upload` — Ingest document
`POST /v1/rag/search` — Perform semantic query
