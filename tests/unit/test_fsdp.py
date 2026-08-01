import torch.nn as nn

from engine.distributed.strategy.fsdp import FSDPStrategy


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


def test_fsdp_strategy_instantiation() -> None:
    strategy = FSDPStrategy()
    assert strategy.device is not None

    model = DummyModel()
    wrapped = strategy.wrap_model(model)
    assert wrapped is not None
