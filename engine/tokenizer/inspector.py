from engine.tokenizer.serialization.io import VocabularySerializer
from engine.tokenizer.vocabulary.inspector import VocabularyInspector


class TokenizerInspector:
    """
    Exposes static inspection methods to audit tokenizer serialization directories.
    """

    @staticmethod
    def inspect_directory(vocab_dir: str) -> None:
        """
        Loads vocabulary configs and prints structured console audits.
        """
        data = VocabularySerializer.load(vocab_dir)
        VocabularyInspector.inspect(data)
