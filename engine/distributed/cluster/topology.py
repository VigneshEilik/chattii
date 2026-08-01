from typing import Any, Dict, List

import torch

from engine.distributed.cluster.node import ClusterNode


class ClusterTopology:
    """Discovers cluster layout including CPUs, GPUs, and node assignments."""

    @staticmethod
    def get_topology() -> Dict[str, Any]:
        node_info = ClusterNode.get_info()
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpus: List[Dict[str, Any]] = []

        for i in range(gpu_count):
            gpus.append(
                {
                    "device_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": torch.cuda.get_device_properties(i).total_memory
                    / (1024**3),
                }
            )

        return {
            "node": node_info,
            "gpu_count": gpu_count,
            "devices": gpus,
        }
