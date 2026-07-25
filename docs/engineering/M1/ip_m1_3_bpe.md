# Chatii – Milestone 1.3: BPE Engine Implementation Plan (IP)

## 1. Objective
Implement a deterministic, production-grade Byte Pair Encoding (BPE) tokenizer training engine, featuring modular pair counting, merge selection, checkpoint saving, validator checking, visualizer outputs, and golden regression testing.

---

## 2. Directory Layout & Files to Build

We will create the following files under `engine/tokenizer/algorithms/bpe/`:
- `__init__.py`
- `pair_counter.py` (counts adjacent token pair frequencies)
- `merge_selector.py` (selects best pair with deterministic tie-breaks)
- `merge_engine.py` (merges adjacent tokens in sequences)
- `merge_history.py` (tracks rank and pair histories)
- `vocabulary_updater.py` (adds new merge tokens to vocabulary manager)
- `checkpoint.py` (checkpoint saving logic)
- `statistics.py` (computes training metrics)
- `visualizer.py` (ASCII growth and compression charts)
- `validator.py` (sanity checks BPE model properties and roundtrips)
- `encoder.py` (greedy subword encoder matching ranks)
- `decoder.py` (converts token IDs back to strings)
- `trainer.py` (iterative BPE training loop orchestrator)
- `algorithm.py` (implements concrete `BPETokenizer` wrapper)

---

## 3. Implementation Checklist

### Step 1: Core Sub-Pipeline Modules
- **`pair_counter.py`**: Count pairs in word sequence frequencies.
- **`merge_selector.py`**: Select highest count pair, resolving ties alphabetically.
- **`merge_engine.py`**: Implement sequence merges replacement logic.
- **`merge_history.py`**: Records mapping of merges.
- **`vocabulary_updater.py`**: Adds merged token strings to `VocabularyManager`.

### Step 2: Training Support Modules
- **`checkpoint.py`**: Saves model registry checkpoints (every 100 merges).
- **`statistics.py`**: Evaluates growth stats.
- **`visualizer.py`**: Format ASCII charts showing vocab growth curves.
- **`validator.py`**: Asserts roundtrip match and duplicate checker.

### Step 3: Trainer, Encoder & Decoder
- **`trainer.py`**: Run character extraction, pretokenize corpus, loop count-select-merge, call checkpointer, return finalized tables.
- **`encoder.py`**: Implements greedy subword merges using loaded merge ranks.
- **`decoder.py`**: Reconstructs text from token IDs.
- **`algorithm.py`**: Implements `BPETokenizer` wrapper complying with `BaseTokenizer`.

### Step 4: Golden Fixtures & Stress Testing
- Create regression assets under `tests/fixtures/tokenizer/`.
- Add stress tests covering mixed-languages (Unicode, Emojis, English, Tamil, Japanese, Chinese) to check roundtrip integrity.

---

## 4. Verification Plan

### Automated Tests
- Run `pytest tests/unit/test_bpe_engine.py` validating:
  - Stream trainers.
  - Periodic checkpoint saving.
  - Multilingual stress test (emojis, Tamil, Japanese, Chinese).
  - Checksum validation on BPE output directories.
  - Golden regression assertions.
