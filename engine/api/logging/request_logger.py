import logging

logger = logging.getLogger("chatii.request")


class RequestLogger:
    """Utility logging prompt tokens, latency, status, and metadata per request."""

    def log_request(
        self,
        request_id: str,
        model_id: str,
        prompt_len: int,
        gen_len: int,
        latency: float,
        status: str,
    ) -> None:
        logger.info(
            {
                "request_id": request_id,
                "model": model_id,
                "prompt_tokens": prompt_len,
                "generated_tokens": gen_len,
                "latency_seconds": latency,
                "status": status,
            }
        )
