from engine.inference.response import GenerationResponse


def test_generation_response():
    resp = GenerationResponse(
        text="hello",
        tokens=["h", "e", "l", "l", "o"],
        token_ids=[1, 2, 3, 3, 4],
        finish_reason="eos_token",
        prompt_tokens=2,
        generated_tokens=5,
        latency_seconds=0.1,
        tokens_per_second=50.0,
    )
    assert resp.text == "hello"
    assert resp.finish_reason == "eos_token"
    assert resp.tokens_per_second == 50.0
