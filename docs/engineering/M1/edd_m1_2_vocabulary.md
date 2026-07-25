# Chatii – Milestone 1.2: Vocabulary Engine Engineering Design Document (EDD)

## 1. MLOps Artifact Lifecycles & Versioning

A vocabulary is an immutable ML artifact. Modifying it after training breaks upstream embeddings. To prevent corruption and support future incremental updates, the Vocabulary Engine models vocabularies as versioned state machines.

### 1.1 State Machine Lifecycle (`vocabulary/lifecycle.py`)
```
[CREATED] ──► [TRAINING] ──► [VALIDATED] ──► [ACTIVE] ──► [ARCHIVED]
```
- **CREATED**: Manager initialized, special tokens registered. Modifiable.
- **TRAINING**: Builder is currently populating frequencies and parsing loader streams. Modifiable.
- **VALIDATED**: Frequencies finalized, validator rules executed and passed. Read-only.
- **ACTIVE**: Marked ready for model embedding lookups. Read-only.
- **ARCHIVED**: Deprecated or replaced version. Excluded from default factory queries. Read-only.

### 1.2 Provenance & Versioning Metadata
Every artifact save includes a parent registry tracking:
- `version`: Semantic version of the vocabulary (e.g., `1.0.0`).
- `parent_version`: Pointer to parent vocabulary for tracking inheritance/incremental training.
- `builder_version`: Version of the builder module that compiled the vocabulary.
- **Dataset Provenance**:
  - Name and relative file path.
  - SHA-256 hash of the training dataset.
  - File size (in bytes) and encoding.
  - Total number of lines and parsed documents.

---

## 2. Component Architecture & Stubs

The engine splits code into specialized modules to adhere to single-responsibility and open-closed principles:

```
engine/tokenizer/vocabulary/
├── manager.py          # Token-ID maps, freezing locks
├── builder.py          # Accumulates token frequencies from loader
├── validator.py        # Audits maps for integrity
├── statistics.py       # Metrics calculations
├── metadata.py         # Versioning and dataset provenance schemas
├── manifest.py         # Directory metadata catalog
├── inspector.py        # CLI stdout formatting
├── cache.py            # Dataset hashing to skip redundant training runs
├── lifecycle.py        # State transitions
├── exceptions.py       # Custom Vocabulary exceptions
├── types.py            # Shared aliases (Token, TokenId)
└── special_tokens.py   # Special token constants
```

---

## 3. Serialization and Verification Security

To prevent manual edits or accidental corruption in production registries, every serialization writes:
1. `vocab.json`: Key-value token mapping.
2. `metadata.json`: Parameters, state, and versioning.
3. `statistics.json`: Evaluation and quality metrics.
4. `special_tokens.json`: Special label mappings.
5. `manifest.json`: Location and state descriptor.
6. `report.md`: Markdown summary of metrics and distribution.
7. `checksum.sha256`: STANDALONE SHA-256 checksum calculated over all the JSON files inside the folder. If any JSON file is altered, loading will raise a `SerializationError`.
