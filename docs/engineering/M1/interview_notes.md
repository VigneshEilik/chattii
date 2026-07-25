# Interview Notes

This document contains key concepts and comparisons to prepare for tokenizer technical discussions.

## 1. Comparing GPT-2 vs GPT-4 vs LLaMA Tokenizers

| Feature | GPT-2 | LLaMA | GPT-4 (cl100k_base) |
|---|---|---|---|
| **Algorithm** | Byte-Level BPE | SentencePiece BPE | Byte-Level BPE |
| **Vocab Size** | 50,257 | 32,000 | 100,256 |
| **Space Handling** | Byte-level mapping | Meta-symbol `_` | Byte-level mapping |
| **Special Tokens** | `<|endoftext|>` | `<s>`, `</s>`, `<unk>` | Multiple special tokens |

## 2. Core Concepts to Remember

- **Out-of-Vocabulary (OOV)**: Resolved by falling back to character-level (or byte-level) base vocabularies.
- **Merge Rules Registry**: The ranks table is static after training, guiding the greedy merges order.
- **Round-Trip Identity**: Reconstructing the exact input string is mandatory for any production tokenizer.
