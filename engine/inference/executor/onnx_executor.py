from typing import Any

import torch

from engine.inference.executor.base import BaseExecutor


class OnnxExecutor(BaseExecutor):
    """ONNX Runtime serving executor stub."""

    def execute(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Return standard forward stub output
        return self.model(input_ids, **kwargs)
