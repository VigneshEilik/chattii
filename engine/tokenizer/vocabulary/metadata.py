from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VocabularyMetadata(BaseModel):
    """
    Provenance and training configuration metadata recorded when compiling
    a Vocabulary artifact.
    """

    version: str = "1.0.0"
    parent_version: Optional[str] = None
    algorithm: str
    created_at: str = Field(
        default_factory=lambda: (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
    )
    vocab_size: int
    reserved_tokens: List[str]
    normalization: str
    lowercase: bool
    min_frequency: int
    dataset_provenance: Dict[str, Any]
    builder_version: str = "1.0.0"
