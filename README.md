# Chatii

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Chatii** is a production-grade, modular large language model (LLM) framework built entirely from scratch in PyTorch. Inspired by the architectures of **Llama, Qwen, Gemma, DeepSeek, and Mistral**, Chatii implements every component from tokenization to serving — no Hugging Face abstractions, no shortcuts.

Built around an **Operating System (OS) design philosophy**, every component (tokenizer, embedding, attention, training, inference, serving) conforms to stable core interfaces, making them fully replaceable without modifying downstream systems.

---

## ✨ Key Features

| Category | Capabilities |
|---|---|
| **Tokenizer** | BPE engine, Unicode-aware, vocabulary builder, production tokenizer with special tokens |
| **Transformer** | Multi-head & grouped-query attention, RoPE/ALiBi positional encodings, RMSNorm/LayerNorm, SwiGLU/GELU FFN, weight tying |
| **Training** | Mixed-precision (AMP), gradient accumulation, checkpointing, learning rate schedulers, sequence packing |
| **Distributed** | DDP, FSDP, pipeline parallelism, tensor parallelism, multi-GPU launcher |
| **Inference** | KV cache (paged, sliding window, prefix), greedy/beam/sampling decoding, Top-K/Top-P/Min-P/Typical/Mirostat, speculative decoding |
| **Constrained Generation** | JSON schema, regex, grammar, and trie-based constrained decoding |
| **API Serving** | OpenAI-compatible REST API, SSE & WebSocket streaming, JWT + API key auth, rate limiting |
| **Deployment** | Docker (CPU + CUDA), Kubernetes manifests, Helm charts, Nginx gateway |
| **SDKs** | Python and JavaScript client libraries |
| **Observability** | Prometheus metrics, OpenTelemetry tracing, structured JSON logging |

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────┐
                    │     Clients / SDKs / CLI        │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   API Gateway (engine/api/)     │
                    │  REST · SSE · WebSocket         │
                    │  Auth · Rate Limit · Middleware  │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   Inference Engine               │
                    │  Generation Loop · KV Cache      │
                    │  Sampling · Beam Search          │
                    │  Streaming · Sessions            │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │   Transformer Model              │
                    │  Attention · FFN · Norm          │
                    │  Embeddings · Positional Enc.    │
                    └────────────────┬────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
   ┌──────────▼──────────┐ ┌────────▼────────┐ ┌───────────▼──────────┐
   │  Training Engine    │ │   Tokenizer     │ │  Dataset Pipeline    │
   │  DDP · FSDP         │ │   BPE · Vocab   │ │  Load · Clean · Pack │
   │  AMP · Checkpoints  │ │   Encode/Decode │ │  Shard · Validate    │
   └─────────────────────┘ └─────────────────┘ └──────────────────────┘
```

### Core Design Principles

- **Interface Stability**: All contracts defined in `core/interfaces/` — replacing an attention mechanism or tokenizer never breaks the pipeline.
- **Double Registry**: Model Registry (`models/`) manages definitions & checkpoints; Dataset Registry (`datasets/`) handles loading, versioning & validation.
- **Replaceable Engines**: Every `engine/` module can be swapped independently (e.g., FlashAttention for standard attention, PagedKVCache for basic cache).

---

## 📁 Project Structure

```
chatii/
├── core/                    # Stable interfaces & contracts
│   └── interfaces/          # BaseTokenizer, BaseAttention, BaseTransformer, ...
│
├── engine/
│   ├── tokenizer/           # BPE engine, vocabulary builder, production tokenizer
│   ├── transformer/         # Attention, FFN, normalization, embeddings, positional encoding
│   ├── training/            # Training loop, optimizers, schedulers, checkpointing, AMP
│   ├── dataset/             # Data loading, cleaning, validation, packing, sharding
│   ├── distributed/         # DDP, FSDP, pipeline/tensor parallelism, launcher
│   ├── inference/           # Generation loop, KV cache, sampling, beam search, streaming
│   └── api/                 # FastAPI serving, OpenAI-compatible routes, middleware
│
├── deployment/
│   ├── docker/              # Dockerfile.cpu, Dockerfile.cuda
│   ├── compose/             # docker-compose.yaml, docker-compose.prod.yaml
│   ├── kubernetes/          # Full K8s manifests (deployment, service, ingress, HPA, etc.)
│   └── helm/                # Helm chart with templates and values
│
├── sdk/
│   ├── python/              # Python client SDK (chatii_sdk)
│   └── javascript/          # JavaScript client SDK
│
├── cli/                     # Unified CLI (train, serve, benchmark, model management)
├── configs/                 # YAML configurations (training, generation, api, distributed)
├── scripts/                 # Demo & benchmark scripts
├── tests/
│   ├── unit/                # 120+ unit tests
│   └── integration/         # 40+ integration tests
├── docs/                    # Documentation
└── pyproject.toml           # Project metadata & tool configuration
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PyTorch 2.x
- CUDA 12.x (optional, for GPU acceleration)

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

### Training

```bash
# Configure hyperparameters
vim configs/training.yaml

# Start training
python cli/main.py train
```

### Inference (Interactive)

```bash
# Run inference demo
python scripts/inference_demo.py
```

### Serve as API

```bash
# Start OpenAI-compatible API server
python cli/main.py serve --host 0.0.0.0 --port 8000

# Or with uvicorn directly
uvicorn engine.api.api:app --host 0.0.0.0 --port 8000
```

### Use the API

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatii-v1",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatii-v1",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### Python SDK

```python
from sdk.python.chatii_sdk import ChatiiClient

client = ChatiiClient(base_url="http://localhost:8000", api_key="your-api-key")

response = client.chat.completions.create(
    model="chatii-v1",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
)
print(response.choices[0].message.content)
```

---

## 🛠️ CLI Reference

```bash
python cli/main.py <command>
```

| Command | Description |
|---|---|
| `train` | Train a model from `configs/training.yaml` |
| `serve` | Launch OpenAI-compatible API server |
| `generate` | Generate text from a prompt |
| `benchmark` | Run latency, throughput & memory benchmarks |
| `benchmark-api` | Run API endpoint benchmarks |
| `config` | Display current configuration |
| `version` | Display version information |
| `list-models` | List all registered models |
| `load-model` | Load a model into memory |
| `unload-model` | Unload a model from memory |
| `reload-model` | Hot-reload a model |
| `health` | Check API server health |
| `metrics` | Retrieve server metrics |

---

## 🐳 Deployment

### Docker

```bash
# CPU
docker build -f deployment/docker/Dockerfile.cpu -t chatii:cpu .

# CUDA (GPU)
docker build -f deployment/docker/Dockerfile.cuda -t chatii:cuda .

# Docker Compose
docker compose -f deployment/compose/docker-compose.yaml up -d
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f deployment/kubernetes/

# Or use Helm
helm install chatii deployment/helm/
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# With coverage
pytest tests/ --cov=engine --cov-report=html
```

**Current status**: 161 tests passing (unit + integration).

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
| **M9** | Multimodal & RAG Capabilities | 🔜 Planned |
| **M10** | Web Playground & Interactive UI | 🔜 Planned |

---

## 📊 Supported Decoding Strategies

| Strategy | Description |
|---|---|
| Greedy | Argmax token selection |
| Top-K | Sample from top K probable tokens |
| Top-P (Nucleus) | Sample from smallest set with cumulative probability ≥ P |
| Min-P | Dynamic threshold based on max probability |
| Temperature | Scale logits before sampling |
| Typical Sampling | Sample tokens near expected information content |
| Mirostat | Adaptive perplexity-targeting sampling |
| Beam Search | Expand multiple hypotheses in parallel |
| Speculative Decoding | Draft-then-verify acceleration |
| Constrained (JSON/Regex/Grammar) | Force output to match a schema or pattern |

---

## 🤝 Contributing

We welcome contributions! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards, testing requirements, and PR guidelines.

---

## 📄 License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.
