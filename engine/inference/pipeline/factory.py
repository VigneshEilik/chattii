from typing import Any

from engine.inference.pipeline.registry import PipelineRegistry


class PipelineFactory:
    """Factory to return concrete pipeline runners."""

    @staticmethod
    def create(name: str, **kwargs: Any) -> Any:
        pipeline_class = PipelineRegistry.get(name)
        return pipeline_class(**kwargs)
