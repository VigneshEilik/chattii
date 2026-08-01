from engine.inference.cache.prefix_cache import (
    GlobalPrefixCache,
    PromptPrefixCache,
    SessionPrefixCache,
)


def test_prefix_caches():
    p_cache = PromptPrefixCache()
    p_cache.set("system_prompt", "value")
    assert p_cache.get("system_prompt") == "value"

    s_cache = SessionPrefixCache()
    s_cache.set("session_1", "state")
    assert s_cache.get("session_1") == "state"

    g_cache = GlobalPrefixCache()
    g_cache.set("global_1", "info")
    assert g_cache.get("global_1") == "info"
