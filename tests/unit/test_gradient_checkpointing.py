import torch.nn as nn

from engine.distributed.optimization.gradient_checkpointing import (
    GradientCheckpointingManager,
)


class MockLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.checkpointed = False

    def gradient_checkpointing_enable(self) -> None:
        self.checkpointed = True


def test_gradient_checkpointing() -> None:
    model = MockLayer()
    GradientCheckpointingManager.enable(model)
    assert model.checkpointed is True
