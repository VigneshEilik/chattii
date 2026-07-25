from typing import List

from engine.tokenizer.vocabulary.manager import VocabularyManager


class BPEDecoder:
    """
    Reconstructs raw string text from sequences of integer token IDs.
    """

    def __init__(self, vocab_manager: VocabularyManager) -> None:
        self.vocab_manager = vocab_manager

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Translates ID array back to concatenated text.
        """
        tokens = []
        specials = set(self.vocab_manager.get_special_tokens().values())

        for tid in token_ids:
            token = self.vocab_manager.id_to_token(tid)
            if skip_special_tokens and token in specials:
                continue
            tokens.append(token)

        return "".join(tokens)
