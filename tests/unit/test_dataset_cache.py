import os
import tempfile

import numpy as np

from engine.dataset.cache.dataset_cache import DatasetCache
from engine.dataset.cache.fingerprint import DatasetFingerprint
from engine.dataset.cache.tokenizer_cache import TokenizerCache


def test_dataset_cache() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = DatasetCache(cache_dir=temp_dir)
        fingerprint = "test_fingerprint_123"
        token_ids = [1, 2, 3, 4, 5]

        # Verify not present initially
        assert cache.get(fingerprint) is None

        # Cache it
        cache.set(fingerprint, token_ids)

        # Retrieve and verify
        cached = cache.get(fingerprint)
        assert cached is not None
        assert np.array_equal(cached, np.array(token_ids, dtype=np.uint16))


def test_tokenizer_cache() -> None:
    cache = TokenizerCache()
    text = "hello world"
    ids = [12, 34, 56]

    # Verify initial get is None
    assert cache.get_ids(text) is None
    assert cache.get_text(ids) is None

    # Set caches
    cache.set_ids(text, ids)
    cache.set_text(ids, text)

    # Verify retrieval
    assert cache.get_ids(text) == ids
    assert cache.get_text(ids) == text

    # Verify clear
    cache.clear()
    assert cache.get_ids(text) is None


def test_dataset_fingerprint() -> None:
    config = {"batch_size": 16, "learning_rate": 1e-4}
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"raw data")
        temp_name = f.name

    try:
        fp1 = DatasetFingerprint.calculate(config, [temp_name])
        fp2 = DatasetFingerprint.calculate(config, [temp_name])
        assert fp1 == fp2

        # Change config
        config_diff = {"batch_size": 32, "learning_rate": 1e-4}
        fp3 = DatasetFingerprint.calculate(config_diff, [temp_name])
        assert fp1 != fp3
    finally:
        os.remove(temp_name)
