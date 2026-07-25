# Chatii

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Chatii is a highly modular, production-ready language model framework built entirely from scratch in PyTorch. It is designed around the **Operating System (OS) design philosophy**, where every component (Tokenizer, Embedding, Attention, layers, training engine, RAG pipeline) conforms to stable core interfaces, allowing them to be fully replaceable without modifying downstream applications.

---

## 🏗️ Architecture & Core Philosophy

Chatii decouples **Production Interfaces** from **Research Engines** to reduce coupling and allow rapid iteration.

```
                  ┌──────────────────────────────┐
                  │          API / SDK           │
                  └──────────────┬───────────────┘
                                 │
                  ┌──────────────▼───────────────┐
                  │    core/interfaces (Stable)  │
                  │   - BaseTokenizer            │
                  │   - BaseAttention            │
                  │   - BaseTransformer          │
                  └──────────────┬───────────────┘
                                 │
                  ┌──────────────▼───────────────┐
                  │    engine/ (Replaceable)     │
                  │   - BpeTokenizer             │
                  │   - MultiHeadAttention       │
                  │   - FlashAttention           │
                  └──────────────────────────────┘
```

- **Interface Stability**: Defined in `core/interfaces/`, these contracts ensure that replacing an attention mechanism with a faster implementation (like FlashAttention) or replacing a tokenizer doesn't break the model pipeline.
- **Double Registry System**:
  - **Model Registry (`models/`)**: Manages model definitions, checkpoint paths, and hyperparameter configuration mapping.
  - **Dataset Registry (`datasets/`)**: Standardizes dataset loading, versioning, checksum validation, and split management.

---

## 🚀 Installation & Local Setup

### Prerequisites
- Python 3.11
- Docker & Docker Compose (optional, for DB & Redis stack)

### 1. Local Environment Setup
Create a virtual environment and install the dependencies:
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy the development env template and customize it:
```bash
cp .env.example .env
```

### 3. Spin up PostgreSQL and Redis Databases
Use docker compose to start the dependencies stack for development:
```bash
docker compose -f docker-compose.dev.yml up -d
```

---

## 🛠️ Usage Overviews

### Training
Configure hyperparameters in `configs/training.yaml` and run training scripts:
```bash
python scripts/train.py --config configs/training.yaml
```

### Inference
Interact with trained weights via the CLI wrapper:
```bash
python scripts/serve.py
```

### FastAPI API
Launch the OpenAI-compatible HTTP serving engine:
```bash
uvicorn api.main:app --reload
```

---

## 🗺️ Roadmap
Please see [ROADMAP.md](ROADMAP.md) for detailed descriptions of all Milestones:
- **v0.1**: Foundation & Tokenizer (Milestone 0 & 1)
- **v0.2**: Transformer Mathematics (Milestone 2)
- **v0.3**: Training Engine (Milestone 3)
- **v0.4**: Inference & Cache Optimization (Milestone 4)
- **v1.0**: Production API, SDK & Interactive Web Playground (Milestones 5 & 6)
- **v2.0**: Multimodality & RAG Capabilities (Milestones 7, 8, 9 & 10)

---

## 🤝 Contributing
We welcome contributions to Chatii! Please review our [CONTRIBUTING.md](CONTRIBUTING.md) to understand coding standards, testing requirements, and how to submit pull requests.

---

## 📄 License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
