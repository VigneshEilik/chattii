# Chatii – Milestone 1: Tokenizer Engine Engineering Design Document (EDD)

## 1. Objective & Scope

The objective of Milestone 1 is to build a production-ready, highly modular tokenization framework for Chatii. The tokenizer acts as the boundary between raw text and the language model's numerical representations.

### In Scope
- Abstract interfaces for Tokenizers, Vocabularies, and Trainers.
- Normalizer module (Unicode forms, whitespace normalization, lowercase conversion).
- Pretokenizer module (regex-based word, character, and custom splitting).
- Vocabulary Manager (mapping tokens to IDs and IDs to tokens, special token reservations, unknown token fallback).
- Character Tokenizer, Word Tokenizer, and Byte Pair Encoding (BPE) implementations.
- Serialization/Deserialization (saving merges, vocabulary maps, and config).
- Factory pattern for instantiating different tokenizer types.
- CLI subcommands for training, encoding, decoding, and inspecting tokenizers.
- Externalized parameters loaded via `configs/tokenizer.yaml`.

### Out of Scope
- Tensor embeddings, Rotary Position Embeddings (RoPE), self-attention math, model training/inference loops, dataset loaders, or API completions (reserved for future milestones).

---

## 2. Theoretical Architecture

A tokenizer transforms raw text into a sequence of token IDs through a multi-stage sequential pipeline, and decodes IDs back into readable text:

```
Encoding Flow:
[Raw Text] -> [Normalizer] -> [Pretokenizer] -> [BPE Algorithm] -> [Token IDs]

Decoding Flow:
[Token IDs] -> [Vocabulary Lookup] -> [Byte/String Merging] -> [Decoded Text]
```

### Byte-Pair Encoding (BPE) Algorithm
1. **Vocabulary Initialization**: Initialize the vocabulary with all unique base units (typically single characters or bytes) and special tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`, `<mask>`).
2. **Frequency Counting**: Given a corpus pretokenized into words, count co-occurrences of adjacent token pairs.
3. **Merge Generation**: Select the most frequent adjacent token pair and merge them into a new single token.
4. **Iterative Refinement**: Repeat step 2 and 3 until the vocabulary reaches the target `vocab_size` or no more frequent pairs exist.
5. **Merge Registry**: Save the sequence of merges (the "merges rules" registry) to resolve out-of-vocabulary splits deterministically at inference/encoding time.

---

## 3. Modular Interface Definitions

To achieve the "Operating System" design philosophy, we define concrete abstractions in `core/interfaces/`:

```python
class BaseVocabulary(ABC):
    @abstractmethod
    def add_token(self, token: str) -> int: ...
    @abstractmethod
    def token_to_id(self, token: str) -> int: ...
    @abstractmethod
    def id_to_token(self, token_id: int) -> str: ...
    @abstractmethod
    def __len__(self) -> int: ...

class BaseTrainer(ABC):
    @abstractmethod
    def train(self, corpus_path: str, vocab_size: int) -> tuple[dict[str, int], list[tuple[str, str]]]: ...

class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    @abstractmethod
    def decode(self, ids: list[int], clean_up_tokenization_spaces: bool = True) -> str: ...
    @abstractmethod
    def save(self, directory: str) -> None: ...
    @abstractmethod
    def load(self, directory: str) -> None: ...
```

---

## 4. Engineering Trade-Offs

### Flat BPE vs. Byte-Level BPE (BBPE)
- **Flat BPE**: Falls back to `<unk>` when encountering characters not present in the training set (e.g., emojis or foreign scripts). This can lead to out-of-vocabulary (OOV) errors during downstream serving.
- **Byte-Level BPE (BBPE)**: Converts raw strings to UTF-8 bytes before applying BPE merging rules. This ensures a vocabulary size of at least 256 base bytes, meaning the model can represent *any* input string without relying on `<unk>`, but increases average sequence length for non-ASCII text.
- **Decision**: We will implement standard BPE targeting string characters for Milestone 1 to establish clean math, but design the pretokenization pipeline to support byte-level mapping if we want to migrate to BBPE in a future ADR.

---

## 5. Acceptance Criteria

Milestone 1 is complete only if:
1. Standard unit tests pass for Character, Word, and BPE tokenizers.
2. An encoding -> decoding round-trip on arbitrary ASCII and UTF-8 strings preserves content identically (excluding normalizations like lowercasing).
3. The BPE trainer yields deterministic vocabularies and merges files from a given seed corpus.
4. Serialized configs can be re-loaded, recreating identical tokenization behavior.
5. The project passes all `ruff` checks and code formatting.
