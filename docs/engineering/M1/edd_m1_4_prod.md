# Chatii – Milestone 1.4: Production Tokenizer Engineering Design Document (EDD)

## 1. Objective
Introduce a modular, algorithm-agnostic registry, service orchestration layer, local configuration loading, health validator, and diagnostic CLI hooks to deliver a clean production tokenizer engine interface.

---

## 2. Structural Design

```
                     configs/tokenizer.yaml
                               │
                               ▼
        TokenizerService  ◄─── TokenizerConfig
               │
               ├──────► TokenizerFactory.create(...)
               │                 │
               │                 ▼
               │          TokenizerRegistry
               │                 │
               │                 ├─► CharacterTokenizer
               │                 ├─► WordTokenizer
               │                 ├─► BPETokenizer
               │                 └─► Stubs (WordPiece, etc.)
               │
               ├──────► Health & Compatibility Checker
               └──────► CLI Main Entrypoint
```

### 2.1 Tokenizer Registry (`registry.py`)
Decouples string identifiers from class objects. Maps algorithm names dynamically. Avoids hardcoded conditional evaluation structures.

### 2.2 Tokenizer Factory (`factory.py`)
Initializes and constructs concrete tokenizers with standard kwargs matching configurations.

### 2.3 Tokenizer Service Layer (`service.py`)
Wraps the active tokenizer instance. The rest of the Chatii model modules (embeddings, attention blocks, etc.) communicate exclusively with this layer.
