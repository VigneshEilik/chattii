from typing import Any, List, Union

import requests

from sdk.python.chatii_sdk.exceptions import ChatiiAPIError


class Embedding:
    """Handles Embeddings calculations."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def create(self, model: str, input_text: Union[str, List[str]]) -> Any:
        url = f"{self.client.base_url}/embeddings"
        headers = {}
        if self.client.api_key:
            headers["Authorization"] = f"Bearer {self.client.api_key}"

        payload = {
            "model": model,
            "input": input_text,
        }

        resp = requests.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise ChatiiAPIError(f"API Error {resp.status_code}: {resp.text}")
        return resp.json()
