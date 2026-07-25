from engine.tokenizer.vocabulary.manager import VocabularyManager


class BPEVocabularyUpdater:
    """
    Handles inserting successfully merged BPE subwords into the VocabularyManager.
    """

    @staticmethod
    def update_vocabulary(
        manager: VocabularyManager, merged_token: str, frequency: int
    ) -> int:
        """
        Inserts the new subword token and logs its frequency count.
        """
        token_id = manager.add_token(merged_token)
        manager._token_frequencies[merged_token] = frequency
        return token_id
