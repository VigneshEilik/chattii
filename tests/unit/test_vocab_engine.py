import json
import shutil
import tempfile
from pathlib import Path

import pytest

from datasets.registry.json_loader import JsonLoader
from datasets.registry.jsonl_loader import JsonlLoader
from datasets.registry.text_loader import TextLoader
from engine.tokenizer.normalizer.unicode import UnicodeNormalizer
from engine.tokenizer.pretokenizer.split import RegexPretokenizer
from engine.tokenizer.serialization.io import VocabularySerializer
from engine.tokenizer.vocabulary.builder import VocabularyBuilder
from engine.tokenizer.vocabulary.exceptions import (
    DuplicateIdError,
    FrozenVocabularyError,
    SerializationError,
)
from engine.tokenizer.vocabulary.lifecycle import VocabularyLifecycle
from engine.tokenizer.vocabulary.manager import VocabularyManager
from engine.tokenizer.vocabulary.statistics import VocabularyStatistics
from engine.tokenizer.vocabulary.validator import VocabularyValidator


def test_loaders() -> None:
    """
    Tests text, json, and jsonl dataset loaders with generator
    streaming and provenance metadata.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. TXT Loader
        txt_file = Path(temp_dir) / "sample.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("line 1\nline 2\n")

        txt_loader = TextLoader()
        lines = list(txt_loader.load_lines(str(txt_file)))
        assert lines == ["line 1\n", "line 2\n"]
        meta = txt_loader.get_provenance_metadata(str(txt_file))
        assert meta["lines_count"] == 2
        assert meta["format"] == "txt"

        # 2. JSON Loader
        json_file = Path(temp_dir) / "sample.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(["json line 1", "json line 2"], f)

        json_loader = JsonLoader()
        json_lines = list(json_loader.load_lines(str(json_file)))
        assert json_lines == ["json line 1", "json line 2"]
        json_meta = json_loader.get_provenance_metadata(str(json_file))
        assert json_meta["lines_count"] == 2
        assert json_meta["format"] == "json"

        # 3. JSONL Loader
        jsonl_file = Path(temp_dir) / "sample.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            f.write('{"text": "jsonl 1"}\n{"text": "jsonl 2"}\n')

        jsonl_loader = JsonlLoader(text_key="text")
        jsonl_lines = list(jsonl_loader.load_lines(str(jsonl_file)))
        assert jsonl_lines == ["jsonl 1", "jsonl 2"]
        jsonl_meta = jsonl_loader.get_provenance_metadata(str(jsonl_file))
        assert jsonl_meta["lines_count"] == 2
        assert jsonl_meta["format"] == "jsonl"

    finally:
        shutil.rmtree(temp_dir)


def test_vocabulary_lifecycle_and_freeze() -> None:
    """
    Tests VocabularyManager state machine transitions and freeze locks.
    """
    manager = VocabularyManager()
    assert manager._lifecycle == VocabularyLifecycle.CREATED
    assert not manager.is_frozen()

    # Add token in created state
    tok_id = manager.add_token("hello")
    assert tok_id >= 5  # Beyond specials

    # Transition training
    manager.transition_to(VocabularyLifecycle.TRAINING)
    manager.add_token("world")
    assert manager.contains("world")

    # Transition validation (freezing)
    manager.freeze()
    assert manager._lifecycle == VocabularyLifecycle.VALIDATED
    assert manager.is_frozen()

    # Assert writes raise errors when frozen
    with pytest.raises(FrozenVocabularyError):
        manager.add_token("blocked")

    with pytest.raises(FrozenVocabularyError):
        manager.remove_token("world")


def test_vocabulary_builder_sorting_and_cache() -> None:
    """
    Tests VocabularyBuilder sorting (frequency then alphabetic) and caching.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        corpus = Path(temp_dir) / "corpus.txt"
        # frequencies: hello=3, world=2, chatii=1, apple=1
        with open(corpus, "w", encoding="utf-8") as f:
            f.write("hello world hello world hello chatii apple")

        normalizer = UnicodeNormalizer()
        pretokenizer = RegexPretokenizer(mode="word")

        builder = VocabularyBuilder(
            normalizer=normalizer,
            pretokenizer=pretokenizer,
            min_frequency=1,
            cache_dir=str(Path(temp_dir) / "cache"),
        )

        # Build first run (cache miss)
        loader = TextLoader()
        vocab = builder.build(str(corpus), loader)
        assert len(vocab) == 10  # 5 specials + 5 unique tokens

        # Verify deterministic ordering:
        # Frequencies: " hello": 2, " world": 2, " apple": 1, " chatii": 1, "hello": 1
        # Order by freq desc, then alphabetical:
        assert vocab.id_to_token(5) == " hello"
        assert vocab.id_to_token(6) == " world"
        assert vocab.id_to_token(7) == " apple"
        assert vocab.id_to_token(8) == " chatii"
        assert vocab.id_to_token(9) == "hello"

        # Build second run (cache hit)
        vocab_cache = builder.build(str(corpus), loader)
        assert len(vocab_cache) == 10

    finally:
        shutil.rmtree(temp_dir)


def test_validator_and_statistics() -> None:
    """
    Tests integrity verification rules and stats computations.
    """
    manager = VocabularyManager()
    manager.transition_to(VocabularyLifecycle.TRAINING)
    manager.add_token("hello")
    manager.add_token("world")
    manager._token_frequencies["hello"] = 10
    manager._token_frequencies["world"] = 5
    manager.freeze()

    # 1. Statistics computation
    stats = VocabularyStatistics.compute(manager)
    assert stats.vocab_size == 7
    assert stats.unique_tokens_count == 2
    assert stats.longest_token == "hello"
    assert stats.longest_token_length == 5
    assert stats.token_entropy > 0.0

    # 2. Validator success
    VocabularyValidator.validate(manager)

    # 3. Validator failure: duplicate token simulation
    corrupt_manager = VocabularyManager()
    corrupt_manager._token_to_id["duplicate"] = 10
    corrupt_manager._token_to_id["another"] = 10
    corrupt_manager._id_to_token[10] = "duplicate"
    corrupt_manager._id_to_token[11] = "another"

    with pytest.raises(DuplicateIdError):
        VocabularyValidator.validate(corrupt_manager)


def test_serialization_and_checksums() -> None:
    """
    Tests saving/loading JSON formats and validation of SHA-256 signatures.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        vocab_map = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "<mask>": 4,
            "hello": 5,
        }
        metadata = {
            "version": "1.0.0",
            "algorithm": "word",
            "vocab_size": 6,
            "reserved_tokens": ["<pad>", "<unk>", "<s>", "</s>", "<mask>"],
            "normalization": "NFKC",
            "lowercase": False,
            "min_frequency": 1,
            "dataset_provenance": {"name": "test"},
            "builder_version": "1.0.0",
        }
        specials = {
            "pad": "<pad>",
            "unk": "<unk>",
            "bos": "<s>",
            "eos": "</s>",
            "mask": "<mask>",
        }
        stats = {
            "vocab_size": 6,
            "reserved_tokens_count": 5,
            "unique_tokens_count": 1,
            "average_token_length": 5.0,
            "longest_token": "hello",
            "shortest_token": "hello",
            "longest_token_length": 5,
            "shortest_token_length": 5,
            "token_entropy": 0.0,
        }
        manifest = {
            "name": "english_v1",
            "algorithm": "word",
            "vocab_size": 6,
            "version": "1.0.0",
            "status": "active",
            "location": str(temp_dir),
        }
        report = "# Vocabulary Report\nSample report content"

        save_dir = Path(temp_dir) / "vocab_registry"

        # Save
        VocabularySerializer.save(
            vocab_dir=str(save_dir),
            vocab_map=vocab_map,
            metadata=metadata,
            special_tokens=specials,
            statistics=stats,
            manifest=manifest,
            report_md=report,
        )

        # Verify existence
        assert (save_dir / "checksum.sha256").exists()
        assert (save_dir / "vocab.json").exists()

        # Load
        loaded = VocabularySerializer.load(str(save_dir))
        assert loaded["vocab"]["hello"] == 5
        assert loaded["metadata"]["algorithm"] == "word"

        # Corrupt file manually and test signature failure
        with open(save_dir / "vocab.json", "w") as f:
            f.write('{"corrupted": true}')

        with pytest.raises(SerializationError):
            VocabularySerializer.load(str(save_dir))

    finally:
        shutil.rmtree(temp_dir)
