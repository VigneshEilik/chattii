from typing import Callable

import torch


class CUDAGraphCapture:
    """Manages static CUDA Graph captures to speed up model execution execution."""

    @staticmethod
    def capture_forward(
        model: torch.nn.Module, example_input: torch.Tensor
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        # Simple CUDA Graph capture stub
        if not example_input.is_cuda:
            return model

        try:
            g = torch.cuda.CUDAGraph()
            # Warmup
            static_input = example_input.clone()
            model(static_input)

            with torch.cuda.graph(g):
                static_output = model(static_input)

            def run_graph(x: torch.Tensor) -> torch.Tensor:
                static_input.copy_(x)
                g.replay()
                return static_output.clone()

            return run_graph
        except Exception:
            return model
