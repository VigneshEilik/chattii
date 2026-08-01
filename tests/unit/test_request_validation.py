import pytest

from engine.api.validation.request_validator import RequestValidator


def test_request_validator():
    validator = RequestValidator(max_prompt_length=100)

    # Valid params should not raise
    validator.validate_generation_params(
        temperature=0.7, top_p=0.9, max_tokens=64, prompt_len=10
    )

    # Invalid temperature
    with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
        validator.validate_generation_params(
            temperature=2.5, top_p=0.9, max_tokens=64, prompt_len=10
        )

    # Invalid prompt length
    with pytest.raises(ValueError, match="exceeds max allowed prompt length"):
        validator.validate_generation_params(
            temperature=0.7, top_p=0.9, max_tokens=64, prompt_len=150
        )
