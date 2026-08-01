import json
from typing import Iterator


class ChatiiStreamResponse:
    """Iterator reading server-sent token events."""

    def __init__(self, response) -> None:
        self.response = response

    def __iter__(self) -> Iterator[dict]:
        for line in self.response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data_str = decoded[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    yield json.loads(data_str)
