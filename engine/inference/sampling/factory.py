from typing import Any

from engine.inference.sampling.registry import BaseSamplingFilter, SamplingRegistry


class SamplingFactory:
    """Factory to create sampling filter instances."""

    @staticmethod
    def create(name: str, **kwargs: Any) -> BaseSamplingFilter:
        filter_class = SamplingRegistry.get(name)
        return filter_class(**kwargs)
