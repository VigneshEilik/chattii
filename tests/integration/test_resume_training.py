import os
import shutil
import tempfile

import torch
import torch.nn as nn

from engine.distributed.checkpoint.distributed import DistributedCheckpointManager
from engine.distributed.checkpoint.restore import DistributedCheckpointRestorer
from engine.distributed.fault_tolerance.recovery import DistributedRecoveryManager


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)


def test_recovery_flow() -> None:
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    temp_dir = tempfile.mkdtemp()
    try:
        checkpoint_path = os.path.join(temp_dir, "step_100")
        DistributedCheckpointManager.save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            {"global_step": 100},
            rank=0,
            world_size=1,
            strategy="ddp",
        )

        latest = DistributedRecoveryManager.find_latest_checkpoint(temp_dir)
        assert latest is not None
        assert "step_100" in latest

        device = torch.device("cpu")
        restored = DistributedCheckpointRestorer.restore(
            latest, model, optimizer, rank=0, device=device
        )
        assert restored["global_step"] == 100
    finally:
        shutil.rmtree(temp_dir)
