from engine.api.state.state import ApplicationState


def execute_shutdown(state: ApplicationState) -> None:
    """Coordinates graceful worker drops and unloads model caches on shutdown."""
    if state.task_dispatcher:
        state.task_dispatcher.stop()
    if state.model_manager:
        state.model_manager.unload_all()
    if state.session_manager:
        state.session_manager.clear()
