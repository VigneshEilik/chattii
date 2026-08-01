import torch.nn as nn

from engine.distributed.strategy.ddp import DDPStrategy


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


def test_ddp_strategy_instantiation() -> None:
    strategy = DDPStrategy()
    assert strategy.device is not None

    model = DummyModel()
    # On single-GPU or CPU fallback it returns or wraps model
    wrapped = strategy.wrap_model(model)
    assert wrapped is not None
