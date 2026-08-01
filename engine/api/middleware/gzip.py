from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware


def setup_gzip(app: FastAPI) -> None:
    """Configures response compression middleware."""
    app.add_middleware(GZipMiddleware, minimum_size=1000)
