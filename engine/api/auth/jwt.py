import base64
import hashlib
import hmac
import json
from typing import Any, Dict, Optional


class JwtAuthenticator:
    """Verifies HS256 JWT tokens natively without requiring external pip dependencies."""

    def __init__(self, secret_key: str, algorithm: str = "HS256") -> None:
        self.secret_key = secret_key
        self.algorithm = algorithm

    def encode(self, payload: Dict[str, Any]) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        header_b = (
            base64.urlsafe_b64encode(json.dumps(header).encode("utf-8"))
            .decode("utf-8")
            .replace("=", "")
        )
        payload_b = (
            base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8"))
            .decode("utf-8")
            .replace("=", "")
        )
        key = self.secret_key.encode("utf-8")
        msg = f"{header_b}.{payload_b}".encode("utf-8")
        sig = hmac.new(key, msg, hashlib.sha256).digest()
        signature_b = base64.urlsafe_b64encode(sig).decode("utf-8").replace("=", "")
        return f"{header_b}.{payload_b}.{signature_b}"

    def _base64url_decode(self, payload: str) -> bytes:
        rem = len(payload) % 4
        if rem > 0:
            payload += "=" * (4 - rem)
        return base64.urlsafe_b64decode(payload)

    def decode_and_verify(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_segment, payload_segment, signature_segment = parts

            # Verify signature
            key = self.secret_key.encode("utf-8")
            msg = f"{header_segment}.{payload_segment}".encode("utf-8")
            sig = hmac.new(key, msg, hashlib.sha256).digest()
            expected_signature = (
                base64.urlsafe_b64encode(sig).decode("utf-8").replace("=", "")
            )

            # Since sigs can have trailing padding differences, check without padding
            if not hmac.compare_digest(
                signature_segment.replace("=", ""), expected_signature
            ):
                return None

            payload_bytes = self._base64url_decode(payload_segment)
            return json.loads(payload_bytes.decode("utf-8"))
        except Exception:
            return None

    def authenticate(self, auth_header: Optional[str]) -> Optional[Dict[str, Any]]:
        if not auth_header:
            return None
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        return self.decode_and_verify(parts[1])
