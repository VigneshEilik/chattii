import torch

from engine.transformer.embedding.factory import EmbeddingFactory
from engine.transformer.embedding.lm_head import LMHead
from engine.transformer.embedding.token import TokenEmbedding
from engine.transformer.embedding.validator import EmbeddingValidator


def test_token_embedding() -> None:
    vocab_size = 1000
    hidden_size = 64
    layer = TokenEmbedding(vocab_size, hidden_size, pad_id=0)

    assert layer.weight.shape == (vocab_size, hidden_size)

    # Input sequence with negative & out-of-bounds IDs
    x = torch.randint(0, vocab_size, (2, 10))
    x[0, 0] = -1
    x[0, 1] = vocab_size + 10

    out = layer(x)
    assert out.shape == (2, 10, hidden_size)

    # Padding elements should be zeroed out
    pad_in = torch.zeros(1, 5, dtype=torch.long)
    pad_out = layer(pad_in)
    assert torch.all(pad_out == 0.0)


def test_lm_head() -> None:
    vocab_size = 1000
    hidden_size = 64
    layer = LMHead(vocab_size, hidden_size, bias=True)
    assert layer.weight.shape == (vocab_size, hidden_size)
    assert layer.bias is not None
    assert layer.bias.shape == (vocab_size,)

    x = torch.randn(2, 10, hidden_size)
    out = layer(x)
    assert out.shape == (2, 10, vocab_size)


def test_weight_tying_and_validation() -> None:
    vocab_size = 1000
    hidden_size = 64

    token_emb = EmbeddingFactory.create(
        "token", vocab_size=vocab_size, hidden_size=hidden_size
    )
    lm_head = EmbeddingFactory.create(
        "lm_head",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        tied_weight=token_emb.weight,
    )

    # Run checks
    ok, errs = EmbeddingValidator.validate_weight_tying(token_emb, lm_head)
    assert ok
    assert len(errs) == 0
