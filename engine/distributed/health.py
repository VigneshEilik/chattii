from typing import Any, Dict

import torch.distributed as dist


class ClusterHealthChecker:
    """Verifies that all nodes in the process group can communicate."""

    @staticmethod
    def verify_health() -> Dict[str, Any]:
        if not dist.is_initialized():
            return {"healthy": True, "note": "Single-process execution"}

        try:
            # Run barrier to ensure all nodes respond
            dist.barrier()
            return {
                "healthy": True,
                "world_size": dist.get_world_size(),
                "backend": dist.get_backend(),
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
            }
