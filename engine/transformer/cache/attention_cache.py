from typing import Dict

from engine.transformer.cache.kv_cache import KVCache


class AttentionCache:
    """
    Maintains a registry of KVCache storage cells indexed by layer number.
    """

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.caches: Dict[int, KVCache] = {i: KVCache() for i in range(num_layers)}

    def get_cache(self, layer_idx: int) -> KVCache:
        """
        Retrieves the Key-Value cache instance for the designated layer.
        """
        if layer_idx not in self.caches:
            self.caches[layer_idx] = KVCache()
        return self.caches[layer_idx]

    def reset(self) -> None:
        """
        Resets and clears the cached histories across all layers.
        """
        for cache in self.caches.values():
            cache.reset()
