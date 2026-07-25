import torch

from engine.transformer.attention.k_projection import KeyProjection
from engine.transformer.attention.output_projection import OutputProjection
from engine.transformer.attention.q_projection import QueryProjection
from engine.transformer.attention.scores import AttentionScores
from engine.transformer.attention.softmax import AttentionSoftmax
from engine.transformer.attention.v_projection import ValueProjection


def test_decoupled_projections() -> None:
    d_model = 64
    num_heads = 4
    head_dim = 16

    q_proj = QueryProjection(d_model, num_heads, head_dim)
    k_proj = KeyProjection(d_model, num_heads, head_dim)
    v_proj = ValueProjection(d_model, num_heads, head_dim)
    out_proj = OutputProjection(d_model, num_heads, head_dim)

    x = torch.randn(2, 10, d_model)

    q = q_proj(x)
    k = k_proj(x)
    v = v_proj(x)

    assert q.shape == (2, 10, num_heads, head_dim)
    assert k.shape == (2, 10, num_heads, head_dim)
    assert v.shape == (2, 10, num_heads, head_dim)

    h_combined = torch.randn(2, 10, num_heads * head_dim)
    out = out_proj(h_combined)
    assert out.shape == (2, 10, d_model)


def test_scores_and_softmax() -> None:
    scores_layer = AttentionScores()
    softmax_layer = AttentionSoftmax(dim=-1)

    q = torch.randn(2, 4, 10, 16)
    k = torch.randn(2, 4, 10, 16)

    scores = scores_layer(q, k)
    assert scores.shape == (2, 4, 10, 10)

    probs = softmax_layer(scores)
    assert probs.shape == (2, 4, 10, 10)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 4, 10), atol=1e-5)
