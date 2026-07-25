# Subsystem Architecture

This document describes the architectural layout of the Chatii Tokenizer subsystem.

```
+-------------------------------------------------------------+
|                       BASE INTERFACES                       |
|                       (BaseTokenizer)                       |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                      TOKENIZER PIPELINE                     |
|                                                             |
|   Unicode Normalizer  ===> Regex Pretokenizer  ===> BPE     |
|   (NFKC, lowercase)        (Words, whitespace)      Engine  |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                      VOCABULARY ENGINE                      |
|                      (vocab.json, merges)                   |
+-------------------------------------------------------------+
```

## Modular Pipeline Flow

1. **Unicode Normalizer**: Converts text into a normalized form (NFKC, NFD, NFC, NFKD).
2. **Regex Pretokenizer**: Splices string inputs into words and whitespace fragments to prevent merges across word boundaries.
3. **BPE Engine**: Performs greedy token combinations using BPE merge ranks to emit final subword IDs.
