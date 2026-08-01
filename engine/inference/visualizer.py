from typing import Any, Dict, List


class InferenceVisualizer:
    """Renders formatted ASCII tables summarizing text throughput and latencies."""

    @staticmethod
    def render_results(results: List[Dict[str, Any]]) -> str:
        lines = [
            "=" * 60,
            f"{'Prompt Len':<12} | {'Gen Len':<10} | {'Latency (s)':<12} | {'Tokens/sec':<10}",
            "-" * 60,
        ]
        for r in results:
            lines.append(
                f"{r.get('prompt_length', 0):<12} | "
                f"{r.get('generation_length', 0):<10} | "
                f"{r.get('latency_seconds', 0.0):<12.3f} | "
                f"{r.get('tps', 0.0):<10.2f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)
