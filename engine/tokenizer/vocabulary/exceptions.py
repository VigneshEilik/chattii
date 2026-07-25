class VocabularyError(Exception):
    """Base exception class for all Chatii vocabulary errors."""

    pass


class VocabularyNotFoundError(VocabularyError):
    """Raised when a requested vocabulary configuration cannot be located."""

    pass


# Alias for backwards compatibility / specs alignment
VocabularyNotFound = VocabularyNotFoundError


class DuplicateTokenError(VocabularyError):
    """Raised when a duplicate token is added to the vocabulary mapping."""

    pass


class DuplicateIdError(VocabularyError):
    """Raised when a duplicate token ID collision occurs."""

    pass


class FrozenVocabularyError(VocabularyError):
    """Raised when modifying a locked (frozen) vocabulary."""

    pass


class SerializationError(VocabularyError):
    """Raised when file IO validation or checksum verification fails."""

    pass
