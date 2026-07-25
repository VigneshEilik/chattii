from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DatasetProvenance:
    """Tracks lineage, licensing, and sourcing information for a dataset."""

    source: str
    license: str = "unknown"
    author: str = "unknown"
    download_url: str = ""
    creation_date: str = ""
    cleaning_version: str = "v1"
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
