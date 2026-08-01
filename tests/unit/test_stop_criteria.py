from engine.inference.runtime.stopping_criteria import (
    CompositeStoppingCriteria,
    StopOnEOS,
    StopOnTokens,
)


def test_stopping_criteria():
    stop_eos = StopOnEOS(eos_token_id=2)
    assert stop_eos([1, 2], last_token=2) is True
    assert stop_eos([1, 3], last_token=3) is False

    stop_tokens = StopOnTokens(max_tokens=3)
    assert stop_tokens([1, 2, 3], last_token=3) is True
    assert stop_tokens([1, 2], last_token=2) is False

    composite = CompositeStoppingCriteria([stop_eos, stop_tokens])
    assert composite([1, 2], last_token=2) is True
    assert composite([1, 2, 3], last_token=4) is True
