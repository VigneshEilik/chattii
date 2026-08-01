import json
import os
from typing import Any, Dict

import torch


class DistributedCheckpointRestorer:
    """Restores sharded parameter values to their target processes on recovery."""

    @staticmethod
    def restore(
        path: str,
        model: torch.nn.Module,
        optimizer: Any,
        rank: int,
        device: torch.device,
    ) -> Dict[str, Any]:
        model_filename = f"model_rank{rank}.pt"
        opt_filename = f"optimizer_rank{rank}.pt"

        # Load sharded state dict
        sharded_model_path = os.path.join(path, model_filename)
        sharded_opt_path = os.path.join(path, opt_filename)

        if not os.path.exists(sharded_model_path) or not os.path.exists(
            sharded_opt_path
        ):
            raise FileNotFoundError(
                f"Distributed checkpoint files missing for rank {rank} under '{path}'"
            )

        sharded_model_state = torch.load(
            sharded_model_path, map_location=device, weights_only=True
        )
        # Load and set state_dict (strict=False since shards are partial)
        model.load_state_dict(sharded_model_state, strict=False)

        sharded_opt_state = torch.load(
            sharded_opt_path, map_location=device, weights_only=True
        )
        optimizer.load_state_dict(sharded_opt_state)

        # Retrieve trainer config on rank 0
        trainer_json = os.path.join(path, "trainer.json")
        if os.path.exists(trainer_json):
            with open(trainer_json, "r", encoding="utf-8") as f:
                return json.load(f)

        return {}
