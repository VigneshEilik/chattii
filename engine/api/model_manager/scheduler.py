from typing import Any


class ModelSwapScheduler:
    """Schedules and manages load queues during multi-model hot swapping operations."""

    def __init__(self, manager: Any) -> None:
        self.manager = manager

    def request_swap(self, model_id: str) -> None:
        # Hot-swapping stub checks memory limits and unloads older models
        if len(self.manager.loaded_models) > 2:
            # Unload oldest model
            oldest_id = next(iter(self.manager.loaded_models))
            self.manager.unload_model(oldest_id)
        self.manager.load_model(model_id)
