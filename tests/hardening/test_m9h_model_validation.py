import torch

from engine.inference.cache.kv_cache import KVCache
from engine.inference.runtime.engine import GenerationEngine
from engine.transformer.config import ModelMathConfig
from engine.transformer.model import ChatiiModel


def test_real_model_instantiation():
    config = ModelMathConfig(
        vocab_size=1000,
        d_model=128,
        num_layers=2,
        n_heads=4,
        max_seq_len=256,
    )
    model = ChatiiModel(config)
    assert model is not None
    params = list(model.parameters())
    assert len(params) > 0


def test_real_model_forward_pass():
    config = ModelMathConfig(
        vocab_size=1000,
        d_model=128,
        num_layers=2,
        n_heads=4,
        max_seq_len=256,
    )
    model = ChatiiModel(config)
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (1, 4, 1000)
    assert not torch.isnan(logits).any()


def test_kv_cache_integration():
    cache = KVCache(num_layers=2)
    assert cache is not None
    assert len(cache.caches) == 2


def test_generation_engine_binding():
    config = ModelMathConfig(
        vocab_size=1000,
        d_model=128,
        num_layers=2,
        n_heads=4,
        max_seq_len=256,
    )
    model = ChatiiModel(config)
    engine = GenerationEngine(model=model)
    assert engine.model == model
