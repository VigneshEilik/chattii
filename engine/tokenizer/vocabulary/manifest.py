from datetime import datetime, timezone

from pydantic import BaseModel, Field


class VocabularyManifest(BaseModel):
    """
    Catalog registry manifest for vocabulary artifacts, mapping the name,
    location, state, and version.
    """

    name: str
    algorithm: str
    vocab_size: int
    version: str = "1.0.0"
    status: str = "active"  # options: "created", "active", "archived"
    created_at: str = Field(
        default_factory=lambda: (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
    )
    location: str
