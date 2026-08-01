import hashlib
import json
import os
from typing import Any, Dict

import torch

from engine.distributed.checkpoint.metadata import DistributedCheckpointMetadata
from engine.distributed.checkpoint.shard import ParameterSharder


class DistributedCheckpointManager:
    """Manages sharded distributed checkpoint serialization and rotation."""

    @staticmethod
    def save_checkpoint(
        path: str,
        model: torch.nn.Module,
        optimizer: Any,
        trainer_state: Dict[str, Any],
        rank: int,
        world_size: int,
        strategy: str,
    ) -> None:
        os.makedirs(path, exist_ok=True)

        # Shard parameter weights
        state_dict = model.state_dict()
        sharded_dict = ParameterSharder.shard_state_dict(state_dict, rank, world_size)

        model_filename = f"model_rank{rank}.pt"
        opt_filename = f"optimizer_rank{rank}.pt"

        torch.save(sharded_dict, os.path.join(path, model_filename))
        torch.save(optimizer.state_dict(), os.path.join(path, opt_filename))

        # Save trainer state on rank 0
        if rank == 0:
            with open(os.path.join(path, "trainer.json"), "w", encoding="utf-8") as f:
                json.dump(trainer_state, f, indent=2)

            # Build files index
            files = {r: f"model_rank{r}.pt" for r in range(world_size)}
            DistributedCheckpointMetadata.write_metadata(
                path, world_size, strategy, files
            )

        # Generate checksum validation directory for each rank
        checksums = {}
        for name in [model_filename, opt_filename]:
            file_path = os.path.join(path, name)
            if os.path.exists(file_path):
                h = hashlib.sha256()
                with open(file_path, "rb") as f_bin:
                    while chunk := f_bin.read(8192):
                        h.update(chunk)
                checksums[name] = h.hexdigest()

        with open(
            os.path.join(path, f"checksum_rank{rank}.json"), "w", encoding="utf-8"
        ) as f_sum:
            json.dump(checksums, f_sum, indent=2)
