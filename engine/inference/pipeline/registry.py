from typing import Any, Callable, Dict, Type


class PipelineRegistry:
    """Manages mappings of model inference pipelines (e.g. Chat vs. Embed)."""

    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[Any]], Type[Any]]:
        def decorator(pipeline_class: Type[Any]) -> Type[Any]:
            cls._registry[name] = pipeline_class
            return pipeline_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[Any]:
        if name not in cls._registry:
            raise KeyError(f"Pipeline '{name}' is not registered.")
        return cls._registry[name]
