from evaluation.runners.benchmark_runner import EvaluationResult


class ReportGenerator:
    """Exports structured markdown reports for benchmark runs."""

    @staticmethod
    def generate_markdown_report(model_name: str, res: EvaluationResult) -> str:
        return f"""# Chatii Evaluation Report

**Model:** {model_name}
**Total Samples:** {res.total_samples}

## Quality Metrics
- **Exact Match Accuracy:** {res.exact_match_acc * 100:.1f}%
- **BLEU Score:** {res.bleu_avg:.3f}
- **ROUGE-L Score:** {res.rouge_avg:.3f}
- **Context Adherence:** {res.context_adherence_avg * 100:.1f}%

## Performance Metrics
- **Time To First Token (TTFT):** {res.avg_ttft_ms:.1f} ms
- **Decode Throughput (TPS):** {res.avg_tps:.1f} tokens/sec
"""
