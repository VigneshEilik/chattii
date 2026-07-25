import json
from pathlib import Path
from typing import Tuple


class TokenizerCompatibilityChecker:
    """
    Validates version compatibility of model configurations and raises migration alerts.
    """

    SUPPORTED_VERSIONS = {"1.0.0"}

    @staticmethod
    def check_compatibility(vocab_dir: str) -> Tuple[bool, str]:
        """
        Parses metadata.json and compares version details.
        """
        path = Path(vocab_dir) / "metadata.json"
        if not path.exists():
            return False, "Missing metadata.json, compatibility cannot be validated."

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            version = data.get("version", "1.0.0")
            if version in TokenizerCompatibilityChecker.SUPPORTED_VERSIONS:
                return True, f"Vocabulary version '{version}' is compatible."
            else:
                return (
                    False,
                    f"Vocabulary version '{version}' is incompatible. "
                    f"Supported: "
                    f"{TokenizerCompatibilityChecker.SUPPORTED_VERSIONS}. "
                    f"Migration is required.",
                )
        except Exception as e:
            return False, f"Failed to parse compatibility metadata: {e}"
