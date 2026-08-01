import os
import shutil
import tempfile

import torch
import torch.nn as nn

from engine.distributed.checkpoint.distributed import DistributedCheckpointManager
from engine.distributed.checkpoint.metadata import DistributedCheckpointMetadata


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)


def test_distributed_checkpoint_indexing() -> None:
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    temp_dir = tempfile.mkdtemp()
    try:
        # Save shard for rank 0 and rank 1
        DistributedCheckpointManager.save_checkpoint(
            temp_dir,
            model,
            optimizer,
            {"step": 10},
            rank=0,
            world_size=2,
            strategy="ddp",
        )
        DistributedCheckpointManager.save_checkpoint(
            temp_dir,
            model,
            optimizer,
            {"step": 10},
            rank=1,
            world_size=2,
            strategy="ddp",
        )

        assert os.path.exists(os.path.join(temp_dir, "model_rank0.pt"))
        assert os.path.exists(os.path.join(temp_dir, "model_rank1.pt"))

        # Verify metadata
        meta = DistributedCheckpointMetadata.read_metadata(temp_dir)
        assert meta["world_size"] == 2
        assert meta["strategy"] == "ddp"
    finally:
        shutil.rmtree(temp_dir)
