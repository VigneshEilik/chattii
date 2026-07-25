# Asymptotic Complexity Analysis

This document analyzes the asymptotic time and space complexity of Chatii's BPE engine.

## 1. Training Complexity

- **Time Complexity**: $O(M \cdot U)$ where:
  - $M$ is the number of merge iterations.
  - $U$ is the number of unique word types in the corpus (not total tokens, since pretokenization frequencies map identical word strings to a single sequence).
- **Space Complexity**: $O(U)$ to store word sequence lists and counts.

## 2. Encoding Complexity

- **Time Complexity**: $O(L^2)$ where $L$ is the length of the word to encode.
  - We scan at most $L-1$ pairs, perform the merge in $O(L)$ time, and repeat at most $L$ times.
- **Space Complexity**: $O(L)$ to keep token sequences.

## 3. Decoding Complexity

- **Time Complexity**: $O(T)$ where $T$ is the number of token IDs.
  - It does $T$ dictionary lookups and joins the resulting strings in $O(T)$ time.
- **Space Complexity**: $O(T)$ to accumulate token strings.
