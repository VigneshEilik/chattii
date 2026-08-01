class PrometheusMetricsTracker:
    """Manages internal metric counters and reports Prometheus-compatible logs."""

    def __init__(self) -> None:
        self.active_requests = 0
        self.total_requests = 0
        self.latency_sum = 0.0
        self.prompt_tokens_sum = 0
        self.completion_tokens_sum = 0
        self.queue_depth = 0
        self.cache_hits = 0

    def increment_active_requests(self) -> None:
        self.active_requests += 1
        self.total_requests += 1

    def decrement_active_requests(self) -> None:
        self.active_requests = max(0, self.active_requests - 1)

    def record_request_latency(self, latency: float) -> None:
        self.latency_sum += latency

    def record_tokens(self, prompt: int, completion: int) -> None:
        self.prompt_tokens_sum += prompt
        self.completion_tokens_sum += completion

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def update_queue_depth(self, depth: int) -> None:
        self.queue_depth = depth
