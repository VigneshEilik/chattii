import shutil
import tempfile
from pathlib import Path

import pytest

from datasets.registry.text_loader import TextLoader
from engine.tokenizer.algorithms.bpe.algorithm import BPETokenizer
from engine.tokenizer.algorithms.bpe.checkpoint import BPECheckpointer
from engine.tokenizer.algorithms.bpe.exceptions import BPEError
from engine.tokenizer.algorithms.bpe.merge_engine import BPEMergeEngine
from engine.tokenizer.algorithms.bpe.merge_history import BPEMergeHistory
from engine.tokenizer.algorithms.bpe.merge_selector import BPEMergeSelector
from engine.tokenizer.algorithms.bpe.pair_counter import BPETokenPairCounter
from engine.tokenizer.algorithms.bpe.validator import BPEValidator
from engine.tokenizer.algorithms.bpe.visualizer import BPEVisualizer


def test_bpe_training_components() -> None:
    """
    Tests granular components: pair counter, merge selector, and merge engine.
    """
    # 1. Pair Counter
    word_freqs = {("h", "e", "l", "l", "o"): 5, ("w", "o", "r", "l", "d"): 2}
    counts = BPETokenPairCounter.count_pairs(word_freqs)
    # Expected: ('l', 'l'): 5, ('l', 'o'): 5, ('h', 'e'): 5, ('e', 'l'): 5, etc.
    assert counts[("l", "l")] == 5
    assert counts[("r", "l")] == 2

    # 2. Merge Selector with tie-breaker
    # All have count 5. Alphabetically, ('e', 'l') comes before ('h', 'e'),
    # ('l', 'l'), ('l', 'o').
    best_pair = BPEMergeSelector.select_best_pair(counts)
    assert best_pair == ("e", "l")

    # 3. Merge Engine
    merged_seq = BPEMergeEngine.merge_sequence(("h", "e", "l", "l", "o"), ("e", "l"))
    assert merged_seq == ("h", "el", "l", "o")

    merged_corp = BPEMergeEngine.merge_corpus(word_freqs, ("e", "l"))
    assert ("h", "el", "l", "o") in merged_corp
    assert merged_corp[("h", "el", "l", "o")] == 5


def test_bpe_training_history_and_checkpoint() -> None:
    """
    Tests history tracking and checkpointers.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        history = BPEMergeHistory()
        history.record(1, ("h", "e"), "he", 100)
        assert history.get_history()[0]["merged"] == "he"

        BPECheckpointer.save_checkpoint(
            output_dir=temp_dir,
            merge_count=100,
            vocab_map={"a": 0, "b": 1},
            merges=[("a", "b")],
            statistics={"compression_ratio": 1.5},
        )
        checkpoint_path = Path(temp_dir) / "checkpoints" / "checkpoint_100.json"
        assert checkpoint_path.exists()
    finally:
        shutil.rmtree(temp_dir)


def test_bpe_tokenizer_workflow_and_serialization() -> None:
    """
    Trains BPE, tests roundtrip encoding/decoding, and asserts
    serialization checksum verification.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        corpus = Path(temp_dir) / "corpus.txt"
        with open(corpus, "w", encoding="utf-8") as f:
            f.write("hello world hello chatii apple\n" * 10)

        # Train BPE
        tokenizer = BPETokenizer(
            unicode_normalization="NFKC",
            lowercase=False,
            min_frequency=1,
            max_vocab_size=15,  # Small size to trigger merges
            special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
            checkpoint_interval=2,
        )

        loader = TextLoader()
        tokenizer.train(str(corpus), loader)

        # 1. Roundtrip checks
        original_text = "hello world apple"
        ids = tokenizer.encode(original_text)
        decoded = tokenizer.decode(ids)
        assert decoded == original_text

        # 2. Save
        save_dir = Path(temp_dir) / "bpe_model"
        tokenizer.save(str(save_dir))

        assert (save_dir / "merges.txt").exists()
        assert (save_dir / "checksum.sha256").exists()

        # 3. Load
        loaded_tokenizer = BPETokenizer()
        loaded_tokenizer.load(str(save_dir))

        encoded_ids = loaded_tokenizer.encode(original_text)
        assert loaded_tokenizer.decode(encoded_ids) == original_text

        # 4. Checksum corruption check
        with open(save_dir / "merges.txt", "w") as f:
            f.write("# Corrupted merges file")

        with pytest.raises(BPEError):
            loaded_tokenizer.load(str(save_dir))

    finally:
        shutil.rmtree(temp_dir)


def test_bpe_multilingual_stress_test() -> None:
    """
    Stress tests BPE roundtrip behavior across emojis and mixed multilingual strings.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        corpus = Path(temp_dir) / "corpus.txt"
        multilingual_content = (
            "hello world\n"
            "வணக்கம் உலகம்\n"  # Tamil
            "こんにちは世界\n"  # Japanese
            "你好世界\n"  # Chinese
            "👋🌍🚀🔥\n"  # Emojis
        )
        with open(corpus, "w", encoding="utf-8") as f:
            f.write(multilingual_content * 5)

        tokenizer = BPETokenizer(
            max_vocab_size=50,
            special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
        )
        loader = TextLoader()
        tokenizer.train(str(corpus), loader)

        # Assert roundtrips for each language segment
        test_inputs = [
            "hello world",
            "வணக்கம் உலகம்",
            "こんにちは世界",
            "你好世界",
            "👋🌍🚀🔥",
            "hello வணக்கம் こんにちは 你好 👋",
        ]

        for text in test_inputs:
            BPEValidator.validate_roundtrip(tokenizer, text)

    finally:
        shutil.rmtree(temp_dir)


def test_bpe_visualizer() -> None:
    """
    Verifies ASCII progress line charting is executed without errors.
    """
    x = [0, 10, 20, 30]
    y = [1.0, 1.2, 1.5, 1.9]
    chart = BPEVisualizer.render_chart("Vocab Growth Test", x, y, height=5, width=20)
    assert "Vocab Growth Test" in chart
    assert "0" in chart
