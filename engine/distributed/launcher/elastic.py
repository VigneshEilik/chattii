from typing import Any, Dict


class ElasticLauncherConfig:
    """Configuration placeholder for elastic multi-node fault-tolerant jobs."""

    def __init__(
        self, min_nodes: int = 1, max_nodes: int = 4, max_restarts: int = 3
    ) -> None:
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.max_restarts = max_restarts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "max_restarts": self.max_restarts,
        }
