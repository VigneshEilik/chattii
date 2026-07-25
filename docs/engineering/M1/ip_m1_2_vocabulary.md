# Chatii – Milestone 1.2: Vocabulary Engine Implementation Plan (IP)

## 1. Objective
Build a production-grade, reusable Vocabulary Engine that acts as the canonical source of truth for all tokenizer algorithms in Chatii, incorporating state machines, dataset provenance, caching, checksum verification, and CLI tooling.

---

## 2. Directory Layout & Files to Build

We will create and populate the following new files:

- **Dataset loaders** in `datasets/registry/`:
  - `base_loader.py` (Abstract `BaseTextLoader` class)
  - `text_loader.py` (TXT files parser yielding line generator)
  - `json_loader.py` (JSON string lists parser)
  - `jsonl_loader.py` (JSON Lines parser)
- **Vocabulary engine** in `engine/tokenizer/vocabulary/`:
  - `manager.py` (Freezable vocabulary lookup)
  - `builder.py` (Frequency counting and token register pipeline)
  - `validator.py` (Duplicate and bounds checker)
  - `statistics.py` (Quality statistics analyzer)
  - `metadata.py` (Pydantic model containing version, settings, and provenance)
  - `manifest.py` (Vocabulary manifest descriptor)
  - `inspector.py` (CLI display printer)
  - `cache.py` (Saves hashes and loaded frequencies to avoid rebuilding)
  - `lifecycle.py` (Lifecycle states enum: CREATED, TRAINING, VALIDATED, ACTIVE, ARCHIVED)
  - `exceptions.py` (Vocabulary custom exceptions)
  - `types.py` (Shared type definitions)
  - `special_tokens.py` (Special token constants)
- **Serialization** in `engine/tokenizer/serialization/`:
  - `io.py` (Orchestrates file serialization with checksum verification)
  - `exporter.py` / `importer.py` (Implements target format translations)
- **Benchmarks hooks** in `benchmarks/tokenizer/`:
  - `benchmark_builder.py` (Skeleton benchmark hook)
  - `benchmark_lookup.py` (Skeleton benchmark hook)
  - `benchmark_serialization.py` (Skeleton benchmark hook)

---

## 3. Detailed Step-by-Step Implementation Checklist

### Step 1: Exceptions and Types
- Create custom exceptions (`VocabularyError`, `VocabularyNotFound`, `DuplicateTokenError`, `DuplicateIdError`, `FrozenVocabularyError`, `SerializationError`).
- Define types (`Token = str`, `TokenId = int`, and Pydantic validation structures).

### Step 2: Special Tokens Constants
- Define standard string tokens (`PAD_TOKEN = "<pad>"`, etc.) in `special_tokens.py`.

### Step 3: Dataset Loaders
- Implement `BaseTextLoader` with generator interfaces, and TXT, JSON, and JSONL loader files. Document dataset statistics (file size, encoding, hash, line count).

### Step 4: Lifecycle State Machine
- Implement `VocabularyLifecycle` transitions inside `lifecycle.py` enforcing read-only checks based on state.

### Step 5: Vocabulary Manager
- Implement `VocabularyManager` holding mapping tables, frequency records, lifecycle states, and freezing locks.

### Step 6: Vocabulary Builder & Cache
- Implement `cache.py` which hashes datasets and saves cached vocab frequencies.
- Implement `builder.py` which counts frequencies, filters by minimum frequency, sorts deterministically, and writes to the manager.

### Step 7: Validator and Statistics
- Implement `validator.py` mapping rules validations.
- Implement `statistics.py` calculating token stats, coverage, and histograms.

### Step 8: Serialization & Checksums
- Implement `io.py` with exporter/importer modules. Compute and verify SHA-256 hash files (`checksum.sha256`). Write `report.md` along with the JSON outputs.

### Step 9: Inspector & CLI
- Implement `inspector.py` rendering vocabulary distributions.
- Integrate `inspect-tokenizer` and `validate-tokenizer` commands in `cli/main.py`.

### Step 10: Benchmark Hooks
- Populate standard benchmarking stubs in `benchmarks/tokenizer/`.

---

## 4. Verification Plan

### Automated Tests
- `pytest tests/unit/test_vocab_engine.py` checking:
  - TXT, JSON, JSONL loaders.
  - Frequency builder constraints and deterministic sort orders.
  - Custom exceptions under locked/frozen states.
  - Validation engine errors (duplicate IDs or missing special tokens).
  - SHA-256 checksum failures (corrupting a file manually and checking if loading raises `SerializationError`).
  - Cache hits (avoiding rebuilding when hashes match).

### Manual Tests
- Train a vocabulary on a sample dataset and run:
  ```bash
  python cli/main.py validate-tokenizer --vocab-dir models/registry/tokenizer/default/
  python cli/main.py inspect-tokenizer --vocab-dir models/registry/tokenizer/default/
  ```
