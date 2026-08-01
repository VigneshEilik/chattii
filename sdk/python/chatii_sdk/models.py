from typing import Any

import requests

from sdk.python.chatii_sdk.exceptions import ChatiiAPIError


class Models:
    """Handles query model listings requests."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def list(self) -> Any:
        url = f"{self.client.base_url}/models"
        headers = {}
        if self.client.api_key:
            headers["Authorization"] = f"Bearer {self.client.api_key}"

        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            raise ChatiiAPIError(f"API Error {resp.status_code}: {resp.text}")
        return resp.json()
