from engine.search.provider import WebSearchProvider


def test_web_search_provider_query():
    provider = WebSearchProvider()
    results = provider.search("transformer models", max_results=3)
    assert len(results) > 0
    assert (
        "transformer" in results[0].snippet.lower()
        or "result" in results[0].title.lower()
    )


def test_web_search_max_results():
    provider = WebSearchProvider()
    results = provider.search("python programming", max_results=1)
    assert len(results) <= 1


def test_web_search_result_fields():
    provider = WebSearchProvider()
    res = provider.search("deep learning")[0]
    assert res.title != ""
    assert res.url.startswith("http")
    assert res.snippet != ""
    assert res.score > 0.0


def test_web_search_empty_query():
    provider = WebSearchProvider()
    results = provider.search("")
    assert isinstance(results, list)


def test_web_search_source():
    provider = WebSearchProvider()
    res = provider.search("test")[0]
    assert res.source == "WebSearch"


def test_web_search_score_bound():
    provider = WebSearchProvider()
    res = provider.search("test")[0]
    assert 0.0 <= res.score <= 1.0


def test_web_search_whitespace_query():
    provider = WebSearchProvider()
    results = provider.search("   ")
    assert isinstance(results, list)


def test_web_search_special_chars():
    provider = WebSearchProvider()
    results = provider.search("PyTorch 2.0 & LLM?")
    assert len(results) > 0
