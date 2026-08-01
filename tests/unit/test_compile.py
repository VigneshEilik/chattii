import torch
import torch.nn as nn

from engine.distributed.compile.graph_capture import CUDAGraphCapture
from engine.distributed.compile.optimization import CompilerOptimizationSettings
from engine.distributed.compile.torch_compile import DistributedModelCompiler


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


def test_compilation_and_graph_capture() -> None:
    model = DummyModel()
    compiled = DistributedModelCompiler.compile(model)
    assert compiled is not None

    # Settings verification
    defaults = CompilerOptimizationSettings.get_defaults()
    assert defaults["triton_enabled"] is True

    # Capture check
    x = torch.randn(2, 4)
    run_graph = CUDAGraphCapture.capture_forward(model, x)
    assert run_graph is not None
