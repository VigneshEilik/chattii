from dataclasses import dataclass
from typing import List


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    score: float = 1.0


class WebSearchProvider:
    """Grounding web search provider interface."""

    def search(self, query: str, max_results: int = 3) -> List[SearchResult]:
        results = [
            SearchResult(
                title=f"Search Result for {query}",
                url="https://docs.chatii.ai/search-grounding",
                snippet=f"Latest verified technical specifications and references for '{query}'.",
                source="WebSearch",
                score=0.95,
            )
        ]
        return results[:max_results]


_search_provider = WebSearchProvider()


def get_search_provider() -> WebSearchProvider:
    return _search_provider
