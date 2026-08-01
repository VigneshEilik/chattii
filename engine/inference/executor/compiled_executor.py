from typing import Any

import torch

from engine.inference.executor.base import BaseExecutor


class CompiledExecutor(BaseExecutor):
    """PyTorch 2.0 graph-compiled runner executor."""

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        # Apply compile to accelerate forward steps
        self.compiled_model = torch.compile(model)

    def execute(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.compiled_model(input_ids, **kwargs)
