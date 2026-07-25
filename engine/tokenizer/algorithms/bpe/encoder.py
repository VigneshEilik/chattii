from typing import Dict, List

from engine.tokenizer.algorithms.bpe.types import Pair
from engine.tokenizer.vocabulary.manager import VocabularyManager


class BPEEncoder:
    """
    Implements rank-based greedy subword merging to map raw strings to token lists.
    """

    def __init__(
        self, vocab_manager: VocabularyManager, merges: Dict[Pair, int]
    ) -> None:
        self.vocab_manager = vocab_manager
        self.merges = merges

    def encode_word(self, word: str) -> List[str]:
        """
        Tokenizes a single pre-token by iteratively applying BPE merges of lowest rank.
        """
        if not word:
            return []

        tokens = list(word)

        while len(tokens) > 1:
            best_pair = None
            best_rank = float("inf")

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # Perform the merge across the current word's tokens list
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens
