from typing import Any

import torch

from engine.inference.executor.base import BaseExecutor


class TorchExecutor(BaseExecutor):
    """Standard PyTorch model runner backend executor."""

    def execute(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.model(input_ids, **kwargs)
