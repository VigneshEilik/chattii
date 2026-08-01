from engine.api.schema.openai import ChatMessage, ChatRequest


def test_openai_api_schemas():
    req = ChatRequest(
        model="chatii-v1",
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.8,
        max_tokens=64,
    )
    assert req.model == "chatii-v1"
    assert len(req.messages) == 1
    assert req.messages[0].content == "hello"
    assert req.temperature == 0.8
