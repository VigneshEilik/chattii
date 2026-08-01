from typing import List, Set


class GrammarConstraint:
    """Constrains generation output according to specified Context-Free Grammars (CFG)."""

    def __init__(self, grammar_rules: str) -> None:
        self.grammar_rules = grammar_rules

    def get_allowed_tokens(self, current_ids: List[int], vocab_size: int) -> Set[int]:
        # Return all tokens as allowed by default (stub interface)
        return set(range(vocab_size))
