from typing import Any, List

from engine.training.events.bus import EventBus
from engine.training.events.hooks import TrainingEvents
from engine.training.training_state import TrainingState


class Callback:
    """Base class for all training pipeline lifecycle callbacks."""

    def on_train_start(self, state: TrainingState, **kwargs: Any) -> None:
        pass

    def on_train_end(self, state: TrainingState, **kwargs: Any) -> None:
        pass

    def on_epoch_start(self, state: TrainingState, **kwargs: Any) -> None:
        pass

    def on_epoch_end(self, state: TrainingState, **kwargs: Any) -> None:
        pass

    def on_step_start(self, state: TrainingState, **kwargs: Any) -> None:
        pass

    def on_step_end(self, state: TrainingState, **kwargs: Any) -> None:
        pass

    def on_validation_start(self, state: TrainingState, **kwargs: Any) -> None:
        pass

    def on_validation_end(self, state: TrainingState, **kwargs: Any) -> None:
        pass


class CallbackHandler:
    """Bridges the EventBus notifications to individual Callback hooks."""

    def __init__(self, callbacks: List[Callback], event_bus: EventBus) -> None:
        self.callbacks = callbacks
        self.event_bus = event_bus

        # Register event listeners
        event_bus.subscribe(
            TrainingEvents.TRAIN_START,
            lambda state, **kw: [c.on_train_start(state, **kw) for c in self.callbacks],
        )
        event_bus.subscribe(
            TrainingEvents.TRAIN_END,
            lambda state, **kw: [c.on_train_end(state, **kw) for c in self.callbacks],
        )
        event_bus.subscribe(
            TrainingEvents.EPOCH_START,
            lambda state, **kw: [c.on_epoch_start(state, **kw) for c in self.callbacks],
        )
        event_bus.subscribe(
            TrainingEvents.EPOCH_END,
            lambda state, **kw: [c.on_epoch_end(state, **kw) for c in self.callbacks],
        )
        event_bus.subscribe(
            TrainingEvents.STEP_START,
            lambda state, **kw: [c.on_step_start(state, **kw) for c in self.callbacks],
        )
        event_bus.subscribe(
            TrainingEvents.STEP_END,
            lambda state, **kw: [c.on_step_end(state, **kw) for c in self.callbacks],
        )
        event_bus.subscribe(
            TrainingEvents.VAL_START,
            lambda state, **kw: [
                c.on_validation_start(state, **kw) for c in self.callbacks
            ],
        )
        event_bus.subscribe(
            TrainingEvents.VAL_END,
            lambda state, **kw: [
                c.on_validation_end(state, **kw) for c in self.callbacks
            ],
        )
