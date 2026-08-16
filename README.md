# Chatii

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Chatii** is a production-grade, modular large language model (LLM) framework and end-user AI assistant platform built entirely from scratch in PyTorch. Inspired by the architectures of **Llama, Qwen, Gemma, DeepSeek, and Mistral**, Chatii implements every component from tokenization to serving, Web UI, RAG, tools, safety, and evaluation.

Built around an **Operating System (OS) design philosophy**, every component (tokenizer, embedding, attention, training, inference, serving, platform) conforms to stable core interfaces, making them fully replaceable without modifying downstream systems.

---

## ✨ Key Features

| Category | Capabilities |
|---|---|
| **Web UI & Platform** | ChatGPT-style SPA, dark/light theme, user auth (JWT/RBAC), persistent conversations, developer API keys (`sk-chatii-xxx`), usage tracking |
| **RAG Knowledge** | PDF, TXT, MD, DOCX, CSV, JSON, HTML document parsers, token chunking, in-memory vector store, semantic search, citations |
| **Tools & Function Calling** | Extensible tool execution engine, builtin calculator & datetime tools, parameter validation |
| **Multimodal** | Vision processor (OCR & image analysis), Audio processor (Speech-to-Text & Text-to-Speech) |
| **Web Search & Grounding** | Web search provider abstraction, result normalization, relevance ranking, citation generation |
| **AI Safety & Guardrails** | Prompt injection detection, PII masking (emails, SSNs, credit cards), execution sandboxing |
| **Tokenizer** | BPE engine, Unicode-aware, vocabulary builder, production tokenizer with special tokens |
| **Transformer** | Multi-head & grouped-query attention, RoPE/ALiBi positional encodings, RMSNorm/LayerNorm, SwiGLU/GELU FFN, weight tying |
| **Training** | Mixed-precision (AMP), gradient accumulation, checkpointing, learning rate schedulers, sequence packing |
| **Distributed** | DDP, FSDP, pipeline parallelism, tensor parallelism, multi-GPU launcher |
| **Inference** | KV cache (paged, sliding window, prefix), greedy/beam/sampling decoding, Top-K/Top-P/Min-P/Typical/Mirostat, speculative decoding |
| **API Serving** | OpenAI-compatible REST API, SSE & WebSocket streaming, JWT + API key auth, rate limiting |
| **Admin & Evaluation** | Admin dashboard (CPU/RAM/GPU/VRAM health, P95 latency, TTFT), LLM evaluation framework (BLEU, ROUGE, perplexity, context adherence) |

---

## 🏗️ Architecture

```
                       ┌─────────────────────┐
                       │    Chatii Web UI    │
                       │  (Single-Page App)  │
                       └──────────┬──────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │       API Gateway         │
                    │ Auth / Rate Limit / CORS  │
                    └─────────────┬─────────────┘
                                  │
        ┌─────────────┬───────────┼───────────┬─────────────┬─────────────┐
        │             │           │           │             │             │
        ▼             ▼           ▼           ▼             ▼             ▼
      Chat           RAG        Tools       Vision        Audio        Admin/Usage
     Service       Engine      Engine      Processor    Processor       Services
        │             │           │           │             │             │
        └─────────────┴───────────┼───────────┴─────────────┴─────────────┘
                                  │
                         ┌────────▼────────┐
                         │ GenerationEngine│
                         │   Milestone 7   │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │ Chatii Model    │
                         │ Transformer LLM │
                         └─────────────────┘

     ┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
     │ PostgreSQL/ │ │    Redis    │ │ Vector Store │ │ Object Store │
     │ SQLite DB   │ │ Cache/State │ │ (In-Memory)  │ │ (Local/S3)   │
     └─────────────┘ └─────────────┘ └──────────────┘ └──────────────┘

     ┌────────────────────────────────────────────────┐
     │ Observability: Prometheus + OpenTelemetry      │
     └────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
chatii/
├── apps/
│   └── web/                 # ChatGPT-style Single-Page Web Application (HTML5/CSS3/JS)
│
├── core/                    # Stable interfaces & contracts
│   └── interfaces/          # BaseTokenizer, BaseAttention, BaseTransformer, ...
│
├── engine/
│   ├── platform/            # User Auth, Conversations, API Keys, Usage Tracking
│   ├── rag/                 # Multi-format parsers, chunker, vector store, retriever
│   ├── tools/               # Function calling engine & builtin tools (calculator, datetime)
│   ├── multimodal/          # Vision (OCR/analysis) & Audio (STT/TTS) processors
│   ├── search/              # Web search grounding & snippet ranking
│   ├── safety/              # Prompt injection detection & PII masking guardrails
│   ├── tokenizer/           # BPE engine, vocabulary builder, production tokenizer
│   ├── transformer/         # Attention, FFN, normalization, embeddings, positional encoding
│   ├── training/            # Training loop, optimizers, schedulers, checkpointing, AMP
│   ├── dataset/             # Data loading, cleaning, validation, packing, sharding
│   ├── distributed/         # DDP, FSDP, pipeline/tensor parallelism, launcher
│   ├── inference/           # Generation loop, KV cache, sampling, beam search, streaming
│   └── api/                 # FastAPI serving, OpenAI-compatible routes, middleware
│
├── evaluation/              # LLM evaluation framework (perplexity, BLEU, ROUGE, reports)
├── deployment/              # Docker, Docker Compose, Kubernetes, Helm, Nginx
├── sdk/                     # Python & JavaScript client SDKs
├── cli/                     # Unified CLI (train, serve, benchmark, model management)
├── configs/                 # YAML configurations
├── scripts/                 # Demo & benchmark scripts
├── tests/
│   ├── unit/                # 210+ unit tests
│   └── integration/         # 50+ integration tests
└── pyproject.toml           # Project configuration
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/VigneshEilik/chattii.git
cd chattii

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Run the Web Platform

```bash
# Start API server and Web UI
python cli/main.py serve --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/app` in your browser to access the ChatGPT-style web interface!

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=engine --cov-report=html
```

**Current status**: **268 tests passing** (unit + integration + E2E platform tests).

---

## 🗺️ Milestone Roadmap

| Milestone | Description | Status |
|---|---|---|
| **M0** | Project Foundation & Scaffolding | ✅ Complete |
| **M1** | BPE Tokenizer Engine | ✅ Complete |
| **M2** | Transformer Mathematics (Attention, FFN, Embeddings, Norms) | ✅ Complete |
| **M3** | Model Assembly & Architecture | ✅ Complete |
| **M4** | Training Engine (AMP, Gradient Accumulation, Checkpointing) | ✅ Complete |
| **M5** | Dataset Pipeline (Loading, Cleaning, Packing, Sharding) | ✅ Complete |
| **M6** | Distributed Training & Scaling (DDP, FSDP, Parallelism) | ✅ Complete |
| **M7** | Inference Engine & Text Generation (KV Cache, Sampling, Streaming) | ✅ Complete |
| **M8** | API Serving & Production Deployment (OpenAI API, Docker, K8s, SDKs) | ✅ Complete |
| **M9** | AI Assistant Platform & User Experience (Web UI, Auth, RAG, Tools, Multimodal, Safety, Eval) | ✅ Complete |

---

## 🤝 Contributing

We welcome contributions! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
