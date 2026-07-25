# Design Decisions

## 1. Why BPE over WordPiece or Unigram?
- **GPT Alignment**: BPE is the standard algorithm for GPT models (e.g. GPT-2, GPT-3, GPT-4, LLaMA).
- **Subword Coverage**: It builds a deterministic subword vocabulary that covers unseen words cleanly by fallback to character tokens.

## 2. Deterministic Tie-Breaking
During candidate merges, when two or more pair candidates share the same frequency, we resolve ties using lexicographical string comparisons on the constituent subwords. This guarantees that model training remains 100% reproducible.

## 3. Pre-Tokenization Boundaries
Merging is confined within words. Splitting on word/punctuation boundaries prevents merges across punctuation or spaces, maintaining high linguistic semantic value.
