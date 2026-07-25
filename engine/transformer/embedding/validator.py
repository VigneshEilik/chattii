from typing import List, Tuple

from engine.transformer.embedding.lm_head import LMHead
from engine.transformer.embedding.token import TokenEmbedding


class EmbeddingValidator:
    """
    Validates token embedding and language model head properties.
    """

    @staticmethod
    def validate_embedding_layer(
        layer: TokenEmbedding, vocab_size: int, hidden_size: int
    ) -> Tuple[bool, List[str]]:
        """
        Validates TokenEmbedding weight dimensions.
        """
        errors = []
        if layer.weight.shape != (vocab_size, hidden_size):
            errors.append(
                f"Expected TokenEmbedding shape {(vocab_size, hidden_size)}, "
                f"got {tuple(layer.weight.shape)}"
            )
        return len(errors) == 0, errors

    @staticmethod
    def validate_weight_tying(
        token_emb: TokenEmbedding, lm_head: LMHead
    ) -> Tuple[bool, List[str]]:
        """
        Ensures that LMHead weights point to the same memory object as TokenEmbedding.
        """
        errors = []
        if token_emb.weight.data is not lm_head.weight.data:
            errors.append(
                "LMHead weight is not tied to TokenEmbedding "
                "(different underlying memory)."
            )
        return len(errors) == 0, errors
