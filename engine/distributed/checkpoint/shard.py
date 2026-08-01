from typing import Any, Dict

import torch


class ParameterSharder:
    """Helper class to partition/shard state dicts for distributed strategies."""

    @staticmethod
    def shard_state_dict(
        state_dict: Dict[str, Any], rank: int, world_size: int
    ) -> Dict[str, Any]:
        sharded: Dict[str, Any] = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                # Shard tensor along the first dimension (row sharding)
                size = tensor.size(0)
                chunk_size = (size + world_size - 1) // world_size
                start = rank * chunk_size
                end = min(size, (rank + 1) * chunk_size)
                if start < size:
                    sharded[key] = tensor[start:end].clone()
                else:
                    sharded[key] = torch.empty(0, dtype=tensor.dtype)
            else:
                sharded[key] = tensor
        return sharded
