import yaml
from fastapi import FastAPI

from engine.api.lifespan import lifespan
from engine.api.logging.json_logger import setup_json_logging
from engine.api.middleware.auth import AuthMiddleware
from engine.api.middleware.cors import setup_cors
from engine.api.middleware.exception import ExceptionMiddleware
from engine.api.middleware.gzip import setup_gzip
from engine.api.middleware.logging import LoggingMiddleware
from engine.api.middleware.metrics import MetricsMiddleware
from engine.api.middleware.rate_limit import RateLimitMiddleware
from engine.api.middleware.request_id import RequestIdMiddleware
from engine.api.router import router as main_router

# Setup Structured JSON logs
setup_json_logging()

# Instantiate FastAPI application
app = FastAPI(
    title="Chatii Inference Serving Platform",
    description="Production-grade OpenAI-compatible LLM serving gateway.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Load configuration dict for setup
try:
    with open("configs/api.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
except Exception:
    cfg = {}

# Set up CORS allowances
setup_cors(app, cfg)

# Set up Gzip compression
setup_gzip(app)

# Register custom middleware layers (ordered from outermost to innermost)
app.add_middleware(ExceptionMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIdMiddleware)

# Include main router paths
app.include_router(main_router)
