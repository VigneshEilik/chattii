from engine.search.provider import get_search_provider


def test_grounded_web_search_resolution():
    provider = get_search_provider()
    results = provider.search("Chatii LLM architecture", max_results=2)
    assert len(results) > 0
    assert results[0].url.startswith("http")
    assert results[0].score > 0.0
    assert results[0].source == "WebSearch"
