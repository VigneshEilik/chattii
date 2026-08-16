from evaluation.reports.report_generator import ReportGenerator
from evaluation.runners.benchmark_runner import BenchmarkRunner, EvaluationSample


def test_production_evaluation_benchmark_run():
    runner = BenchmarkRunner()
    samples = [
        EvaluationSample(
            prompt="What is gradient descent?",
            target="Gradient descent is an optimization algorithm.",
            context="Gradient descent minimizes loss by following negative gradients.",
        ),
        EvaluationSample(
            prompt="What is 10 + 20?",
            target="30",
            context="10 + 20 = 30",
        ),
    ]

    res = runner.run_evaluation(samples)
    assert res.total_samples == 2
    assert res.exact_match_acc == 1.0
    assert res.bleu_avg > 0.0
    assert res.rouge_avg > 0.0
    assert res.context_adherence_avg > 0.0

    report = ReportGenerator.generate_markdown_report("chatii-v1", res)
    assert "Exact Match Accuracy" in report
    assert "Context Adherence" in report
