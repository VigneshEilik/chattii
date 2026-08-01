from typing import Any

from engine.distributed.strategy.registry import BaseStrategy, StrategyRegistry


class StrategyFactory:
    """Factory to create strategy instances based on config or names."""

    @staticmethod
    def create(name: str, **kwargs: Any) -> BaseStrategy:
        strategy_class = StrategyRegistry.get(name)
        return strategy_class(**kwargs)
