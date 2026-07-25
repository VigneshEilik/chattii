# Future Extensions

## 1. WordPiece Integration
WordPiece merges pairs by maximizing likelihood (using a language model objective) rather than raw pair frequency. The modular `BaseTokenizer` allows adding WordPiece as a separate algorithm registry option.

## 2. SentencePiece Integration
SentencePiece treats input as a raw byte stream (handling spaces as a meta-symbol e.g. `_`). Future revisions can implement this stream pretokenization strategy.

## 3. Byte-Level BPE (BBPE)
BBPE represents base vocabularies as 256 bytes instead of characters. This prevents Out-Of-Vocabulary (OOV) tokens, allowing complete coverage of any text sequence.
