from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DatasetMetadata:
    """Metadata describing dataset specifications."""

    name: str
    version: str = "v1"
    format: str = "txt"
    sample_count: int = 0
    token_count: int = 0
    size_bytes: int = 0
    sha256_fingerprint: str = ""
    created_at: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
