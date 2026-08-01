import torch

from engine.inference.cache.kv_cache import KVCache
from engine.inference.cache.manager import CacheManager


def test_standard_kv_cache():
    cache = KVCache(num_layers=2)
    # Layer 0 cache
    l0 = cache.get_cache(0)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    rk, rv = l0.update(k, v)
    assert rk.shape == (1, 2, 4, 8)

    # Append along seq dimension
    k2 = torch.randn(1, 2, 1, 8)
    v2 = torch.randn(1, 2, 1, 8)
    rk2, rv2 = l0.update(k2, v2)
    assert rk2.shape == (1, 2, 5, 8)


def test_cache_manager():
    mgr = CacheManager(max_cached_sessions=2)
    mgr.allocate("s1", num_layers=4)
    mgr.allocate("s2", num_layers=4)
    assert len(mgr.cached_sessions) == 2

    # FIFO eviction check
    mgr.allocate("s3", num_layers=4)
    assert "s1" not in mgr.cached_sessions
    assert len(mgr.cached_sessions) == 2
