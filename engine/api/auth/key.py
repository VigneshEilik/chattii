from typing import List, Optional


class KeyAuthenticator:
    """Verifies static Bearer API keys against the whitelist configuration."""

    def __init__(self, allowed_keys: List[str]) -> None:
        self.allowed_keys = allowed_keys

    def authenticate(self, auth_header: Optional[str]) -> bool:
        if not auth_header:
            return False

        # Expect 'Bearer <key>'
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False

        token = parts[1]
        return token in self.allowed_keys
