# Algorithm Deep-Dive

## Byte Pair Encoding (BPE)

Byte Pair Encoding (BPE) is a bottom-up subword tokenization algorithm.

### 1. Training Algorithm
1. Initialize the vocabulary with all unique characters occurring in the corpus.
2. Tokenize the entire corpus into sequences of characters.
3. Count all adjacent character pairs.
4. Select the most frequent pair. Tie-break using a deterministic alphabetical rule.
5. Merge this pair to form a new subword token. Add it to the vocabulary.
6. Replace all occurrences of this pair in the tokenized corpus.
7. Repeat steps 3-6 until the target vocabulary size is reached or no more merges can be made.

### 2. Encoding Algorithm
Encoding uses a greedy rank-based lookup:
1. Deconstruct word into characters.
2. Identify all adjacent pairs.
3. Look up the rank of each pair in the merges table.
4. Merge the pair with the lowest rank (highest priority).
5. Repeat until no adjacent pairs in the word exist in the merges table.
