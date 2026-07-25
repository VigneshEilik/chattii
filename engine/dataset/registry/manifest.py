import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class DatasetManifest:
    """Canonical dataset descriptor storing statistical profiles and hashes."""

    dataset_name: str
    version: str = "v1"
    creation_date: str = ""
    tokenizer_hash: str = ""
    token_count: int = 0
    sample_count: int = 0
    sequence_length: int = 2048
    language: str = "en"
    split_info: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetManifest":
        data = json.loads(json_str)
        return cls(**data)
