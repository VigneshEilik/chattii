import torch

from engine.inference.sampling.registry import BaseSamplingFilter, SamplingRegistry


@SamplingRegistry.register("temperature")
class TemperatureFilter(BaseSamplingFilter):
    """Scales logit shapes to influence decoding randomness."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(1e-6, temperature)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature == 1.0:
            return logits
        return logits / self.temperature
