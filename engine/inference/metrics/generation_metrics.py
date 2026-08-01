from typing import List


class ServingMetricsTracker:
    """Tracks latency percentiles, throughput (TPS), cache stats, and queue depths."""

    def __init__(self) -> None:
        self.ttft_records: List[float] = []
        self.latency_records: List[float] = []
        self.prompt_tokens_count = 0
        self.generated_tokens_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def record_request(
        self, ttft: float, latency: float, p_tok: int, g_tok: int
    ) -> None:
        self.ttft_records.append(ttft)
        self.latency_records.append(latency)
        self.prompt_tokens_count += p_tok
        self.generated_tokens_count += g_tok

    def record_cache(self, hit: bool) -> None:
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    @property
    def average_ttft(self) -> float:
        return (
            sum(self.ttft_records) / len(self.ttft_records)
            if self.ttft_records
            else 0.0
        )

    @property
    def p95_latency(self) -> float:
        if not self.latency_records:
            return 0.0
        sorted_records = sorted(self.latency_records)
        idx = int(len(sorted_records) * 0.95)
        return sorted_records[min(idx, len(sorted_records) - 1)]

    @property
    def p99_latency(self) -> float:
        if not self.latency_records:
            return 0.0
        sorted_records = sorted(self.latency_records)
        idx = int(len(sorted_records) * 0.99)
        return sorted_records[min(idx, len(sorted_records) - 1)]

    @property
    def prompt_tps(self) -> float:
        # Prompt tokens per second prefill stub
        return 0.0

    @property
    def decode_tps(self) -> float:
        # Generation decode tokens per second stub
        return 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
