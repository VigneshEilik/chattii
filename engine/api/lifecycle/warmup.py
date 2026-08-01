from engine.api.state.state import ApplicationState


def execute_warmup(state: ApplicationState) -> None:
    """Evaluates warmup model passes on newly loaded weights."""
    if state.model_manager:
        state.model_manager.warmup_models()
