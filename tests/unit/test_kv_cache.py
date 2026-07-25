import torch

from engine.transformer.cache.attention_cache import AttentionCache
from engine.transformer.config import ModelMathConfig
from engine.transformer.model import ChatiiModel


def test_kv_cache_mechanism() -> None:
    config = ModelMathConfig(vocab_size=1000, d_model=64, n_heads=4, num_layers=2)
    model = ChatiiModel(config)

    # 1. Prefill pass: seq_len = 5
    x_prefill = torch.randint(0, 1000, (1, 5))
    cache = AttentionCache(num_layers=2)

    logits_prefill = model(x_prefill, cache=cache, use_cache=True)
    assert logits_prefill.shape == (1, 5, 1000)

    # Check cache contains 5 tokens
    for i in range(2):
        layer_cache = cache.get_cache(i)
        assert layer_cache.k is not None
        assert layer_cache.v is not None
        assert layer_cache.k.shape == (1, 4, 5, 16)

    # 2. Incremental decode pass: seq_len = 1
    x_decode = torch.randint(0, 1000, (1, 1))
    logits_decode = model(x_decode, cache=cache, use_cache=True)
    assert logits_decode.shape == (1, 1, 1000)

    # Check cache is appended (now contains 6 tokens)
    for i in range(2):
        layer_cache = cache.get_cache(i)
        assert layer_cache.k.shape == (1, 4, 6, 16)

    # 3. Clear cache
    cache.reset()
    for i in range(2):
        layer_cache = cache.get_cache(i)
        assert layer_cache.k is None
        assert layer_cache.v is None
