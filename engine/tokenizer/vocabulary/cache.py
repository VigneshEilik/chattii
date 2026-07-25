import json
from pathlib import Path
from typing import Dict, Optional


class VocabularyCache:
    """
    Manages cached frequency dictionaries based on dataset hashes
    to accelerate vocabulary rebuild times.
    """

    def __init__(self, cache_dir: str = "datasets/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_frequencies(self, dataset_hash: str) -> Optional[Dict[str, int]]:
        """
        Retrieves cached frequencies if they exist and are valid.
        """
        cache_file = self.cache_dir / f"freq_{dataset_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def save_frequencies(self, dataset_hash: str, frequencies: Dict[str, int]) -> None:
        """
        Saves token frequencies associated with the dataset hash.
        """
        cache_file = self.cache_dir / f"freq_{dataset_hash}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(frequencies, f)
        except OSError:
            pass
