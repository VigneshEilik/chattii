from engine.inference.prompt.formatter import ChatFormatter
from engine.inference.prompt.template import PromptTemplate


class MockTokenizer:
    def encode(self, text: str) -> list:
        # Mock encode returns character lengths
        return [ord(c) for c in text]


def test_prompt_template():
    tpl = PromptTemplate("chatml")
    messages = [
        {"role": "system", "content": "You are Chatii."},
        {"role": "user", "content": "Hello"},
    ]
    formatted = tpl.format(messages)
    assert "<|im_start|>system" in formatted
    assert "You are Chatii." in formatted
    assert "<|im_start|>user" in formatted
    assert "Hello" in formatted


def test_chat_formatter():
    tok = MockTokenizer()
    fmt = ChatFormatter(tok)
    messages = [{"role": "user", "content": "Hi"}]
    ids = fmt.format_and_tokenize(messages, max_length=20)
    assert len(ids) <= 20
