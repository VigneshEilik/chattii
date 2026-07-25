import os
import shutil
import tempfile
from pathlib import Path

import pytest

from engine.tokenizer.algorithms.character import CharacterTokenizer
from engine.tokenizer.base.tokenizer import BaseTokenizer
from engine.tokenizer.benchmark import TokenizerBenchmark
from engine.tokenizer.compatibility import TokenizerCompatibilityChecker
from engine.tokenizer.config import TokenizerConfig
from engine.tokenizer.factory import TokenizerFactory
from engine.tokenizer.health import TokenizerHealthChecker
from engine.tokenizer.profiler import TokenizerProfiler
from engine.tokenizer.registry import StubTokenizer, TokenizerRegistry
from engine.tokenizer.service import TokenizerService
from engine.tokenizer.visualizer import TokenizerVisualizer


def test_tokenizer_registry_and_factory() -> None:
    """
    Asserts that registry dynamically registers new classes, maps
    algorithms, and builds stubs.
    """
    # 1. Base registered types
    assert TokenizerRegistry.get("bpe").__name__ == "BPETokenizer"
    assert TokenizerRegistry.get("character") == CharacterTokenizer

    # 2. Dynamic registration
    class CustomDummy(BaseTokenizer):
        def train(self, corpus_path: str) -> None:
            pass

        def encode(self, text: str, add_special_tokens: bool = True):
            return []

        def decode(self, ids):
            return ""

        def save(self, path: str) -> None:
            pass

        def load(self, path: str) -> None:
            pass

        def get_vocab_size(self) -> int:
            return 42

        def get_special_tokens(self):
            return {}

    TokenizerRegistry.register("dummy", CustomDummy)
    assert TokenizerRegistry.get("dummy") == CustomDummy

    dummy_inst = TokenizerFactory.create("dummy")
    assert dummy_inst.get_vocab_size() == 42

    # 3. Future stubs
    stub_cls = TokenizerRegistry.get("sentencepiece")
    assert issubclass(stub_cls, StubTokenizer)

    stub_inst = TokenizerFactory.create("sentencepiece")
    with pytest.raises(NotImplementedError):
        stub_inst.train("corpus.txt")


def test_tokenizer_service_and_config() -> None:
    """
    Tests unified configuration file loader and wrapper service dispatch actions.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        yaml_path = Path(temp_dir) / "test_config.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("algorithm: character\nvocab_size: 250\n")

        config = TokenizerConfig.load_from_yaml(str(yaml_path))
        assert config.algorithm == "character"
        assert config.vocab_size == 250

        # Load default fallback
        fallback = TokenizerConfig.load_from_yaml("nonexistent.yaml")
        assert fallback.algorithm == "bpe"

        # Service wrappers
        service = TokenizerService(config.algorithm)
        assert service.get_vocab_size() == 5  # Standard special tokens count

    finally:
        shutil.rmtree(temp_dir)


def test_health_compatibility_profiler_and_benchmark() -> None:
    """
    Verifies checker diagnostics, execution profiler, and benchmark
    throughput measurements.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        corpus = Path(temp_dir) / "corpus.txt"
        with open(corpus, "w", encoding="utf-8") as f:
            f.write("chatii tokenization production services layers\n" * 10)

        # Train BPE via Service Layer
        service = TokenizerService("bpe", max_vocab_size=30)
        service.train(str(corpus))

        save_dir = Path(temp_dir) / "model_out"
        service.save(str(save_dir))

        # 1. Health checker verification
        healthy, errors = TokenizerHealthChecker.check_health(str(save_dir), "bpe")
        assert healthy, f"Expected healthy model, got: {errors}"

        # Test unhealthy detection
        os.remove(save_dir / "vocab.json")
        unhealthy, fail_errors = TokenizerHealthChecker.check_health(
            str(save_dir), "bpe"
        )
        assert not unhealthy
        assert any("Missing required file: vocab.json" in err for err in fail_errors)

        # Re-save to repair
        service.save(str(save_dir))

        # 2. Compatibility checker
        ok, msg = TokenizerCompatibilityChecker.check_compatibility(str(save_dir))
        assert ok
        assert "compatible" in msg

        # 3. Profiler runs
        prof = TokenizerProfiler.profile_action(service.encode, "chatii layers")
        assert "elapsed_seconds" in prof
        assert "ram_peak_bytes" in prof

        # 4. Benchmark runs
        metrics = TokenizerBenchmark.run(
            service.tokenizer, ["chatii layers"], iterations=10
        )
        assert metrics["iterations"] == 10
        assert "encode_throughput_tokens_per_sec" in metrics

        # 5. Visualizations ASCII
        lengths_chart = TokenizerVisualizer.render_length_distribution(str(save_dir))
        assert "Token Length Distribution" in lengths_chart

        merge_chart = TokenizerVisualizer.render_merge_history(str(save_dir))
        assert "BPE Merge Frequencies" in merge_chart

    finally:
        shutil.rmtree(temp_dir)
