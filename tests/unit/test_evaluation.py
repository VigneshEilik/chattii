from evaluation.metrics.generation import GenerationMetrics
from evaluation.reports.report_generator import ReportGenerator
from evaluation.runners.benchmark_runner import BenchmarkRunner, EvaluationSample


def test_exact_match_metric():
    assert GenerationMetrics.exact_match("hello", "hello") == 1.0
    assert GenerationMetrics.exact_match("hello", "world") == 0.0


def test_bleu_metric():
    score = GenerationMetrics.bleu_score("the cat sat on mat", "the cat sat on the mat")
    assert score > 0.5


def test_perplexity_metric():
    ppl = GenerationMetrics.perplexity(0.0)
    assert round(ppl, 2) == 1.0


def test_context_adherence_metric():
    adh = GenerationMetrics.context_adherence("cat sat", "the cat sat on the mat")
    assert adh == 1.0


def test_benchmark_runner_empty():
    runner = BenchmarkRunner()
    res = runner.run_evaluation([])
    assert res.total_samples == 0


def test_benchmark_runner_samples():
    runner = BenchmarkRunner()
    samples = [
        EvaluationSample(
            prompt="What is AI?",
            target="AI is artificial intelligence",
            context="AI is artificial intelligence",
        ),
        EvaluationSample(prompt="2+2?", target="4", context="2+2=4"),
    ]
    res = runner.run_evaluation(samples)
    assert res.total_samples == 2
    assert res.exact_match_acc == 1.0
    assert res.avg_ttft_ms > 0.0
    assert res.avg_tps > 0.0


def test_report_generator():
    runner = BenchmarkRunner()
    samples = [EvaluationSample(prompt="Q", target="A")]
    res = runner.run_evaluation(samples)
    report = ReportGenerator.generate_markdown_report("chatii-v1", res)
    assert "# Chatii Evaluation Report" in report
    assert "chatii-v1" in report


def test_context_adherence_zero_prediction():
    assert GenerationMetrics.context_adherence("", "some context") == 0.0


def test_token_overlap_zero_target():
    assert GenerationMetrics.token_overlap("pred", "") == 0.0


def test_rouge_metric():
    score = GenerationMetrics.rouge_l("test", "test")
    assert score == 1.0
