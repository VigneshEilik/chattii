import hashlib
import secrets


def hash_password(password: str) -> str:
    """Hashes a raw password using PBKDF2 with SHA-256 and a random salt."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
    )
    return f"{salt}${key.hex()}"


def verify_password(password: str, hashed_password: str) -> bool:
    """Verifies a raw password against a stored salt$hash string."""
    try:
        salt, stored_key = hashed_password.split("$")
        key = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
        )
        return secrets.compare_digest(key.hex(), stored_key)
    except Exception:
        return False
