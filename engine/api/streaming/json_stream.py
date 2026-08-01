import json
from typing import Iterator


def format_json_stream(token_generator: Iterator[str]) -> Iterator[str]:
    """Formats generator tokens into a stream of JSON lines."""
    for token in token_generator:
        yield json.dumps({"token": token}) + "\n"
