# Interview Q&As

## Q1: Why do we pretokenize text instead of training BPE directly on raw text?
**Answer**: Pretokenization splits text into words/whitespace fragments using regex patterns. This prevents merges from spanning across spaces or punctuation. Without pretokenization, a common word followed by punctuation (like `the.`) might get merged, reducing the semantic usefulness and compositionality of the vocabulary.

## Q2: How does BPE ensure that we never encounter Out-Of-Vocabulary (OOV) tokens?
**Answer**: By including all possible individual characters (or all 256 raw bytes) in the initial base vocabulary. Because any unseen word can always be decomposed into its constituent characters/bytes, the tokenizer can always fallback to representing it using these base tokens.

## Q3: What is the time complexity of the BPE greedy encoding process?
**Answer**: If a word is $L$ characters long, there are at most $L-1$ adjacent pairs. In the worst case, each merge step takes $O(L)$ to traverse the token array and replace the pair. We can do at most $L$ merges, leading to a worst-case complexity of $O(L^2)$ per word.
