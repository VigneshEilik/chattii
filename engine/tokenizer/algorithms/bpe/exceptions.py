from engine.tokenizer.vocabulary.exceptions import VocabularyError


class BPEError(VocabularyError):
    """Base exception for BPE algorithm and trainer issues."""

    pass


class BPETrainingError(BPEError):
    """Raised during training runtime violations."""

    pass


class BPEEncodingError(BPEError):
    """Raised when encoding rules mapping fails."""

    pass


class BPEDecodingError(BPEError):
    """Raised when decoding token indexes back to string fails."""

    pass
