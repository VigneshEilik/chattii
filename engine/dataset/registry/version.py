import re


class DatasetVersion:
    """Manages semantic dataset versions (e.g. v1, TinyStories-v2)."""

    def __init__(self, version_str: str = "v1") -> None:
        self.version_str = version_str.strip()

    def is_valid(self) -> bool:
        return bool(re.match(r"^[a-zA-Z0-9_\-\.]+$", self.version_str))

    def __str__(self) -> str:
        return self.version_str
