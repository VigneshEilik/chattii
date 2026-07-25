# Chatii Development Roadmap

Development milestones are grouped into semantic release versions.

---

## v0.1 — Foundation & Tokenizer
- [x] **Milestone 0: Project Skeleton**
  - Unified multi-environment structure
  - Pydantic configuration loader with environment overrides
  - Standard JSON structured logger
  - Dev/Prod Docker & Docker Compose setup
  - CI/CD workflow placeholders
- [ ] **Milestone 1: Pluggable Tokenizers**
  - Character and word tokenizers
  - Custom Byte-Pair Encoding (BPE) tokenizer implementation
  - Pre-tokenization and post-tokenization pipelines
  - Vocab serialization and decoding logic

---

## v0.2 — Transformer Mathematics
- [ ] **Milestone 2: Transformer Layers from Scratch**
  - Modular tensor embedding layer
  - Rotary Position Embeddings (RoPE)
  - Multi-Head Attention (MHA) and Grouped-Query Attention (GQA)
  - RMSNorm layer
  - SwiGLU activation block
  - DecoderBlock layer and language model head

---

## v0.3 — Training Engine
- [ ] **Milestone 3: Training & Checkpoint Pipelines**
  - Pluggable dataset loader and batching utility
  - Cross-entropy loss computation
  - AdamW optimizer and cosine schedule implementation
  - Auto-resume and checkpoint save states
  - CUDA execution and mixed precision utilities

---

## v0.4 — Optimized Inference
- [ ] **Milestone 4: Autoregressive Decoding & KV Caching**
  - Greedy decoding, Top-k, and Top-p (nucleus) sampling
  - Temperature adjustments
  - Key-Value (KV) cache for fast autoregressive generation
  - Interactive CLI application

---

## v1.0 — API Serving & Playground
- [ ] **Milestone 5: OpenAI-Compatible REST API**
  - FastAPI serving endpoints (`POST /v1/chat/completions`, `GET /v1/models`)
  - Streaming tokens response format (Server-Sent Events)
  - API key verification and rate limiting middleware
- [ ] **Milestone 6: Web Playground Interface**
  - Rich chat interface with Markdown rendering and code block styling
  - Streaming token responses and conversation history management

---

## v2.0 — Extended Platform Capabilities
- [ ] **Milestone 7: Retrieval-Augmented Generation (RAG)**
  - Document parser and chunking pipelines
  - Vector db wrappers (FAISS / Qdrant)
- [ ] **Milestone 8: Multimodality**
  - Vision encoder, audio encoder, speech-to-text, and text-to-speech blocks
- [ ] **Milestone 9: Tool Calling**
  - Calculator, Python sandbox, search engine tool integration
- [ ] **Milestone 10: Fine-Tuning Engine**
  - LoRA and QLoRA adapters
- [ ] **Milestone 11: Production Observability**
  - Metrics logging dashboard, usage stats, model registries
