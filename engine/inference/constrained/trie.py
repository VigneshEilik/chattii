from typing import Any, Dict, List, Set


class TrieConstraint:
    """Filters logits using a search trie structure to allow only known vocabulary phrases."""

    def __init__(self, vocabulary_phrases: List[List[int]]) -> None:
        self.trie: Dict[int, Any] = {}
        for phrase in vocabulary_phrases:
            node = self.trie
            for token_id in phrase:
                if token_id not in node:
                    node[token_id] = {}
                node = node[token_id]

    def get_allowed_tokens(self, current_ids: List[int], vocab_size: int) -> Set[int]:
        # Simple trie lookup stub returning all tokens by default
        return set(range(vocab_size))
