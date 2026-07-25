from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class TextSample:
    """Represents a raw or cleaned text sample with optional metadata."""

    text: str
    sample_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenizedSample:
    """Represents a tokenized text sample with input IDs and optional attention mask."""

    input_ids: List[int]
    sample_id: Optional[str] = None
    attention_mask: Optional[List[int]] = None
    labels: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDataset:
    """Abstract base class for streaming datasets."""

    def __iter__(self) -> Iterator[TextSample]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError("Streaming datasets might not implement len().")
