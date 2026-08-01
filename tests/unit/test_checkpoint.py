import os
import shutil
import tempfile

import torch
import torch.nn as nn

from engine.distributed.checkpoint.distributed import DistributedCheckpointManager
from engine.distributed.checkpoint.restore import DistributedCheckpointRestorer


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)


def test_distributed_checkpointing() -> None:
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer_state = {"global_step": 42}

    temp_dir = tempfile.mkdtemp()
    try:
        # Save rank 0 state
        DistributedCheckpointManager.save_checkpoint(
            temp_dir,
            model,
            optimizer,
            trainer_state,
            rank=0,
            world_size=1,
            strategy="ddp",
        )
        assert os.path.exists(os.path.join(temp_dir, "model_rank0.pt"))
        assert os.path.exists(os.path.join(temp_dir, "metadata.json"))

        # Restore
        device = torch.device("cpu")
        restored_state = DistributedCheckpointRestorer.restore(
            temp_dir, model, optimizer, rank=0, device=device
        )
        assert restored_state["global_step"] == 42
    finally:
        shutil.rmtree(temp_dir)
