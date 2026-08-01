import logging

logger = logging.getLogger("chatii.access")


class AccessLogger:
    """Utility logging standard ASGI server access events."""

    def log_access(
        self, method: str, path: str, status_code: int, latency: float
    ) -> None:
        logger.info(
            {
                "method": method,
                "path": path,
                "status_code": status_code,
                "latency": latency,
            }
        )
