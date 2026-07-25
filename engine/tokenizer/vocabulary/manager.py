from typing import Dict, List, Optional

from engine.tokenizer.base.vocabulary import BaseVocabulary
from engine.tokenizer.vocabulary.exceptions import FrozenVocabularyError
from engine.tokenizer.vocabulary.lifecycle import VocabularyLifecycle


class VocabularyManager(BaseVocabulary):
    """
    Production-grade Vocabulary Manager.
    Acts as the canonical lookup for Token-ID conversions and tracks
    lifecycle lock states.
    """

    def __init__(self, special_tokens: Optional[List[str]] = None) -> None:
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._token_frequencies: Dict[str, int] = {}
        self._special_tokens: Dict[str, str] = {}
        self._lifecycle: VocabularyLifecycle = VocabularyLifecycle.CREATED

        # Default special tokens
        specials = special_tokens or ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
        for token in specials:
            self.add_token(token)

        mapping = {
            "<pad>": "pad",
            "<unk>": "unk",
            "<s>": "bos",
            "</s>": "eos",
            "<mask>": "mask",
        }
        for token, name in mapping.items():
            if token in specials:
                self._special_tokens[name] = token

    def transition_to(self, new_state: VocabularyLifecycle) -> None:
        """
        Transitions the lifecycle state of the vocabulary.
        """
        if not self._lifecycle.can_transition_to(new_state):
            raise ValueError(
                f"Invalid lifecycle transition from {self._lifecycle.name} "
                f"to {new_state.name}."
            )
        self._lifecycle = new_state

    def add_token(self, token: str) -> int:
        """
        Registers a string token. Raises FrozenVocabularyError if vocabulary is locked.
        """
        if self.is_frozen():
            raise FrozenVocabularyError(
                f"Vocabulary is frozen in state '{self._lifecycle.name}' "
                f"and cannot be modified."
            )

        if token not in self._token_to_id:
            token_id = len(self._token_to_id)
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
            self._token_frequencies[token] = 0
            return token_id
        return self._token_to_id[token]

    def remove_token(self, token: str) -> None:
        """
        Removes a token. Re-indexes remaining items to preserve zero-indexed continuity.
        """
        if self.is_frozen():
            raise FrozenVocabularyError(
                f"Vocabulary is frozen in state '{self._lifecycle.name}' "
                f"and cannot be modified."
            )

        if token not in self._token_to_id:
            return

        self._token_to_id.pop(token)
        self._token_frequencies.pop(token, None)

        # Re-index all tokens continuously
        remaining_tokens = list(self._token_to_id.keys())
        self._token_to_id.clear()
        self._id_to_token.clear()

        for idx, tok in enumerate(remaining_tokens):
            self._token_to_id[tok] = idx
            self._id_to_token[idx] = tok

    def contains(self, token: str) -> bool:
        """
        Returns True if the token exists.
        """
        return token in self._token_to_id

    def get_id(self, token: str) -> int:
        """
        Wrapper around token_to_id.
        """
        return self.token_to_id(token)

    def get_token(self, token_id: int) -> str:
        """
        Wrapper around id_to_token.
        """
        return self.id_to_token(token_id)

    def freeze(self) -> None:
        """
        Transitions state to VALIDATED, locking mutations.
        """
        if self._lifecycle == VocabularyLifecycle.CREATED:
            self.transition_to(VocabularyLifecycle.TRAINING)

        if self._lifecycle == VocabularyLifecycle.TRAINING:
            self.transition_to(VocabularyLifecycle.VALIDATED)

    def is_frozen(self) -> bool:
        """
        Checks if the current lifecycle state prevents modification.
        """
        return self._lifecycle in {
            VocabularyLifecycle.VALIDATED,
            VocabularyLifecycle.ACTIVE,
            VocabularyLifecycle.ARCHIVED,
        }

    # --- BaseVocabulary Compliance ---

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        unk_token = self._special_tokens.get("unk", "<unk>")
        return self._token_to_id.get(unk_token, 0)

    def id_to_token(self, token_id: int) -> str:
        if token_id in self._id_to_token:
            return self._id_to_token[token_id]
        return self._special_tokens.get("unk", "<unk>")

    def get_special_tokens(self) -> Dict[str, str]:
        return self._special_tokens

    def __len__(self) -> int:
        return len(self._token_to_id)
