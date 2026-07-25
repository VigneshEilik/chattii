import json
import os
from typing import Any, Dict, List, Optional


class DatasetRegistry:
    """Manages the persistence and mapping of registered training datasets."""

    def __init__(self, registry_file: str = "datasets/registry.json") -> None:
        self.registry_file = registry_file
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    self.datasets = json.load(f)
            except Exception:
                self.datasets = {}
        else:
            # Seed default directories and registry
            os.makedirs(os.path.dirname(self.registry_file) or ".", exist_ok=True)
            self.datasets = {
                "tinystories": {
                    "name": "tinystories",
                    "version": "v1",
                    "path": "datasets/demo/tiny.txt",
                    "format": "txt",
                    "language": "en"
                }
            }
            self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.registry_file) or ".", exist_ok=True)
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.datasets, f, indent=2)

    def register(self, name: str, config: Dict[str, Any]) -> None:
        self.datasets[name] = config
        self.save()

    def remove(self, name: str) -> bool:
        if name in self.datasets:
            del self.datasets[name]
            self.save()
            return True
        return False

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self.datasets.get(name)

    def list_all(self) -> List[Dict[str, Any]]:
        return list(self.datasets.values())
