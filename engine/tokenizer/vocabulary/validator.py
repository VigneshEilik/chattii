from engine.tokenizer.vocabulary.exceptions import (
    DuplicateIdError,
    DuplicateTokenError,
    VocabularyError,
)
from engine.tokenizer.vocabulary.manager import VocabularyManager


class VocabularyValidator:
    """
    Performs verification audits on vocabulary structures to prevent index
    and representation corruptions.
    """

    @staticmethod
    def validate(manager: VocabularyManager) -> None:
        """
        Runs complete pre-flight assertions. Raises VocabularyError
        subclasses on failure.
        """
        if len(manager) == 0:
            raise VocabularyError("Vocabulary mapping table is empty.")

        # Check bijective cardinality
        num_tokens = len(manager._token_to_id)
        num_ids = len(manager._id_to_token)

        if num_tokens != num_ids:
            raise VocabularyError(
                f"Vocabulary maps cardial asymmetry. "
                f"Tokens size = {num_tokens}, IDs size = {num_ids}."
            )

        # Check for duplicates
        unique_tokens = set(manager._token_to_id.keys())
        unique_ids = set(manager._token_to_id.values())

        if len(unique_tokens) != num_tokens:
            raise DuplicateTokenError(
                "Vocabulary registry contains duplicate string entries."
            )
        if len(unique_ids) != num_tokens:
            raise DuplicateIdError(
                "Vocabulary registry contains duplicate integer ID mappings."
            )

        # Check range bounds: [0, vocab_size)
        for token_id in unique_ids:
            if token_id < 0 or token_id >= len(manager):
                raise DuplicateIdError(
                    f"Token ID {token_id} falls outside valid range "
                    f"[0, {len(manager)})."
                )

        # Check reciprocal correspondence symmetry
        for token, token_id in manager._token_to_id.items():
            if manager._id_to_token.get(token_id) != token:
                raise VocabularyError(
                    f"Reciprocal map failure: Token '{token}' resolves to {token_id}, "
                    f"but ID {token_id} maps to '{manager._id_to_token.get(token_id)}'."
                )

        # Verify critical special tokens exist
        specials = manager.get_special_tokens()
        if "unk" not in specials:
            raise VocabularyError(
                "Vocabulary missing required '<unk>' fallback token mapping."
            )
