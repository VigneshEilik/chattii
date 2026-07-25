from engine.transformer.config import ModelMathConfig
from engine.transformer.factory import (
    ActivationFactory,
    NormalizationFactory,
    PositionFactory,
)
from engine.transformer.nn.activation import GELU
from engine.transformer.nn.normalization import RMSNorm
from engine.transformer.service import TransformerMathService


def test_factories_instantiation() -> None:
    act = ActivationFactory.create("gelu")
    assert isinstance(act, GELU)

    norm = NormalizationFactory.create("rmsnorm", dim=32)
    assert isinstance(norm, RMSNorm)

    rope = PositionFactory.create("rope", dim=16, max_seq_len=128)
    assert rope.dim == 16


def test_math_service() -> None:
    config = ModelMathConfig(
        activation="silu", normalization="layernorm", position_embedding="sinusoidal"
    )
    service = TransformerMathService(config)
    assert service.activation.__class__.__name__ == "SiLU"
    assert service.normalization.__class__.__name__ == "LayerNorm"
