from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI

from engine.api.lifecycle.cleanup import execute_cleanup
from engine.api.lifecycle.shutdown import execute_shutdown
from engine.api.lifecycle.startup import execute_startup
from engine.api.lifecycle.warmup import execute_warmup
from engine.api.state.state import ApplicationState


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Application State
    state = ApplicationState()

    # Load configuration
    try:
        with open("configs/api.yaml", "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except Exception:
        config_dict = {}

    # Run startup hooks
    execute_startup(state, config_dict)
    execute_warmup(state)

    app.state.model_manager = state.model_manager
    app.state.session_manager = state.session_manager
    app.state.metrics_tracker = state.metrics_tracker
    app.state.config = state.config
    app.state.limiter = state.limiter
    app.state.task_dispatcher = state.task_dispatcher

    yield

    # Run shutdown & cleanup hooks
    execute_shutdown(state)
    execute_cleanup(state)
