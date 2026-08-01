from engine.api.model_manager.manager import ModelManager
from engine.api.monitoring.prometheus import PrometheusMetricsTracker
from engine.api.rate_limit.limiter import RateLimiter
from engine.api.session.manager import SessionManager
from engine.api.state.state import ApplicationState
from engine.api.workers.dispatcher import TaskDispatcher


def execute_startup(state: ApplicationState, config_dict: dict) -> None:
    """Coordinates service initializations and DB connections on startup."""
    state.config = config_dict
    state.model_manager = ModelManager(config_dict)
    state.session_manager = SessionManager()
    state.metrics_tracker = PrometheusMetricsTracker()
    state.limiter = RateLimiter(config_dict)
    state.task_dispatcher = TaskDispatcher()

    # Initialize background worker loop
    state.task_dispatcher.start()
