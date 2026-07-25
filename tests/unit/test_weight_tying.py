from engine.transformer.config import ModelMathConfig
from engine.transformer.model import ChatiiModel


def test_explicit_weight_tying() -> None:
    config = ModelMathConfig(vocab_size=1000, d_model=64, num_layers=1)

    # 1. Enabled weight tying
    model_tied = ChatiiModel(config, weight_tying=True)
    assert model_tied.token_embedding.weight.data is model_tied.lm_head.weight.data

    # 2. Disabled weight tying
    model_untied = ChatiiModel(config, weight_tying=False)
    assert (
        model_untied.token_embedding.weight.data is not model_untied.lm_head.weight.data
    )
