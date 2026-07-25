from engine.dataset.base.dataset import TextSample, TokenizedSample
from engine.dataset.statistics.calculator import DatasetStatisticsCalculator
from engine.dataset.reports.report_generator import DatasetReportGenerator
import tempfile
import os


def test_dataset_statistics_calculator() -> None:
    calc = DatasetStatisticsCalculator()

    sample1 = TextSample("Hello world", metadata={"language": "en"})
    token1 = TokenizedSample(input_ids=[10, 20, 2])  # 2 is EOS

    sample2 = TextSample("Short", metadata={"language": "en"})
    token2 = TokenizedSample(input_ids=[5, 2])

    calc.add_sample(sample1, token1)
    calc.add_sample(sample2, token2)

    stats = calc.calculate(vocab_size=100)

    assert stats["sample_count"] == 2
    assert stats["token_count"] == 5
    assert stats["char_stats"]["max"] == 11
    assert stats["char_stats"]["min"] == 5
    assert stats["special_tokens"]["eos_frequency"] == 2
    assert stats["languages"]["en"] == 2


def test_dataset_report_generator() -> None:
    stats = {
        "sample_count": 10,
        "token_count": 100,
        "char_stats": {"average": 50.0},
        "token_stats": {"average": 10.0},
        "vocabulary": {
            "unique_tokens_seen": 50,
            "coverage_pct": 50.0,
            "oov_ratio_pct": 0.0,
        },
        "special_tokens": {"eos_frequency": 10, "pad_frequency": 0},
        "padding_pct": 0.0,
        "languages": {"en": 10},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        report_path = os.path.join(temp_dir, "dataset_report.md")
        md = DatasetReportGenerator.generate_report(stats, report_path)
        assert "Total Samples" in md
        assert "10" in md
        assert os.path.exists(report_path)
