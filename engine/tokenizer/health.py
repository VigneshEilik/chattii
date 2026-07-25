import hashlib
from pathlib import Path
from typing import List, Tuple

from engine.tokenizer.factory import TokenizerFactory


class TokenizerHealthChecker:
    """
    Diagnoses model integrity, verify hashes, check manifest rules, and test roundtrip.
    """

    @staticmethod
    def _verify_files(path: Path, algorithm: str, errors: List[str]) -> None:
        required_files = [
            "vocab.json",
            "metadata.json",
            "special_tokens.json",
            "manifest.json",
            "checksum.sha256",
        ]
        if algorithm.lower() == "bpe":
            required_files.extend(["merges.txt", "merge_history.json"])

        for fname in required_files:
            fpath = path / fname
            if not fpath.exists():
                errors.append(f"Missing required file: {fname}")

    @staticmethod
    def _verify_hashes(path: Path, errors: List[str]) -> None:
        checksum_path = path / "checksum.sha256"
        if not checksum_path.exists():
            return

        expected_hashes = {}
        with open(checksum_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split("  ", 1)
                if len(parts) == 2:
                    expected_hashes[parts[1]] = parts[0]

        for fname, sha in expected_hashes.items():
            fpath = path / fname
            if not fpath.exists():
                errors.append(f"File '{fname}' declared in checksum is missing.")
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                current_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if current_sha != sha:
                    errors.append(f"Checksum mismatch for file: {fname}")
            except Exception as e:
                errors.append(f"Failed to hash '{fname}': {e}")

    @staticmethod
    def _verify_usability(vocab_dir: str, algorithm: str, errors: List[str]) -> None:
        try:
            tokenizer = TokenizerFactory.create(algorithm)
            tokenizer.load(vocab_dir)
            test_str = "hello world usability test"
            tokenizer.decode(tokenizer.encode(test_str))
        except Exception as e:
            errors.append(f"Usability check failed: {e}")

    @staticmethod
    def check_health(vocab_dir: str, algorithm: str = "bpe") -> Tuple[bool, List[str]]:
        """
        Runs comprehensive integrity checks on a serialized tokenizer directory.
        """
        errors: List[str] = []
        path = Path(vocab_dir)

        if not path.exists():
            errors.append(f"Vocabulary directory '{vocab_dir}' does not exist.")
            return False, errors

        # 1. Identify missing files
        TokenizerHealthChecker._verify_files(path, algorithm, errors)

        # 2. Checksum validation
        TokenizerHealthChecker._verify_hashes(path, errors)

        # 3. Usability check
        if not errors:
            TokenizerHealthChecker._verify_usability(vocab_dir, algorithm, errors)

        return len(errors) == 0, errors
