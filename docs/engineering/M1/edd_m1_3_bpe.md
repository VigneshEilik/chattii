# Chatii – Milestone 1.3: BPE Engine Engineering Design Document (EDD)

## 1. Context & BPE Training Pipeline Architecture

Modern tokenizers require robust, memory-efficient, and deterministic training engines. This document specifies the sub-components of Chatii's standalone BPE training engine.

```
                  Raw Input Text Corpus
                            │
                            ▼
                    Unicode Normalizer
                            │
                            ▼
                    Regex Pretokenizer
                            │
                            ▼
                    Frequencies Counter
                            │
                            ▼
                 [   BPE Trainer Loop   ]
                 ┌──────────────────────┐
                 │ 1. Pair Counter      │ ◄──── Checkpoint Engine
                 │ 2. Merge Selector    │       (every N iterations)
                 │ 3. Merge Engine      │
                 │ 4. Vocab Updater     │
                 └──────────┬───────────┘
                            │
                            ▼
                     Tokenizer Model
            (vocab.json, merges.txt, metadata)
```

---

## 2. Component Design & API Contracts

### 2.1 Pair Counter (`bpe/pair_counter.py`)
Counts frequencies of adjacent token pairs across the tokenized corpus frequencies dictionary.
- Input: `Dict[Tuple[str, ...], int]` (Tokenized corpus sequences to counts).
- Output: `Dict[Tuple[str, str], int]` (Pair counts).

### 2.2 Merge Selector (`bpe/merge_selector.py`)
Identifies the highest frequency pair. Implements a deterministic tie-breaker:
- Sorts by frequency descending.
- Alphabetically by first token ascending.
- Alphabetically by second token ascending.

### 2.3 Merge Engine (`bpe/merge_engine.py`)
Applies a merge rule `(pair_0, pair_1) ➔ merged` on a sequence of tokens:
```python
def merge_sequence(sequence: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
    """
    Transforms sequence e.g. ('h', 'e', 'l', 'l', 'o') with pair=('e', 'l')
    into ('h', 'el', 'l', 'o').
    """
    pass
```

### 2.4 Checkpoint Engine (`bpe/checkpoint.py`)
Saves training states periodically (e.g., every 100 merges):
- Serializes `checkpoint.json` containing current merge count, active vocabulary, and merges list.
- Enables recovery or step-based vocabulary evaluation.

---

## 3. Algorithmic Specifications & Complexities

### 3.1 Time Complexity
- **Iterative Merges**: For $M$ merges on a corpus of size $N$ tokens:
  - Naive pair counting: $O(M \times N)$
  - With token frequency caching: $O(M \times U)$ where $U$ is the number of unique word types in the vocabulary.

### 3.2 Visualizer Output (`bpe/visualizer.py`)
Prints ASCII terminal charts showing:
- Vocabulary growth (merge index vs. vocab size).
- Compression ratio improvements over training steps.
- Top merges distribution histogram.
