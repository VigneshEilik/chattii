from typing import Callable, Dict


class BackendRegistry:
    """Registry class managing distributed communication backends."""

    _registry: Dict[str, Callable[[], None]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
        def decorator(fn: Callable[[], None]) -> Callable[[], None]:
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable[[], None]:
        if name not in cls._registry:
            raise KeyError(f"Backend '{name}' is not registered.")
        return cls._registry[name]
