from typing import Any, Dict, List

import requests

from sdk.python.chatii_sdk.exceptions import ChatiiAPIError
from sdk.python.chatii_sdk.stream import ChatiiStreamResponse


class Chat:
    """Handles Chat completions requests."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 128,
        stream: bool = False,
    ) -> Any:
        url = f"{self.client.base_url}/chat/completions"
        headers = {}
        if self.client.api_key:
            headers["Authorization"] = f"Bearer {self.client.api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if stream:
            resp = requests.post(url, json=payload, headers=headers, stream=True)
            if resp.status_code != 200:
                raise ChatiiAPIError(f"API Error {resp.status_code}: {resp.text}")
            return ChatiiStreamResponse(resp)

        resp = requests.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            raise ChatiiAPIError(f"API Error {resp.status_code}: {resp.text}")
        return resp.json()
