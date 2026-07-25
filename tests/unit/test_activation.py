import torch

from engine.transformer.nn.activation import GELU, ReLU, Sigmoid, SiLU, Softmax, Tanh


def test_activations_numerical_outputs() -> None:
    x = torch.tensor([-2.0, 0.0, 2.0])

    relu = ReLU()
    assert torch.equal(relu(x), torch.tensor([0.0, 0.0, 2.0]))

    gelu = GELU(approximate="tanh")
    out_gelu = gelu(x)
    assert out_gelu.shape == (3,)

    silu = SiLU()
    out_silu = silu(x)
    assert out_silu.shape == (3,)

    sigmoid = Sigmoid()
    out_sig = sigmoid(x)
    assert torch.all(out_sig >= 0.0) and torch.all(out_sig <= 1.0)

    tanh = Tanh()
    out_tanh = tanh(x)
    assert torch.all(out_tanh >= -1.0) and torch.all(out_tanh <= 1.0)

    softmax = Softmax(dim=-1)
    out_sm = softmax(x)
    assert torch.allclose(out_sm.sum(), torch.tensor(1.0))
