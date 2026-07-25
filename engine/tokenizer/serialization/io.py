import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from engine.tokenizer.vocabulary.exceptions import SerializationError


class VocabularySerializer:
    """
    Handles saving and loading of all five JSON files and report.md,
    verifying file integrity using SHA-256 signatures.
    """

    @staticmethod
    def save(
        vocab_dir: str,
        vocab_map: Dict[str, int],
        metadata: Dict[str, Any],
        special_tokens: Dict[str, str],
        statistics: Dict[str, Any],
        manifest: Dict[str, Any],
        report_md: str,
    ) -> None:
        """
        Serializes all vocabulary files and records checksum signatures.
        """
        path = Path(vocab_dir)
        path.mkdir(parents=True, exist_ok=True)

        files_to_write = {
            "vocab.json": vocab_map,
            "metadata.json": metadata,
            "special_tokens.json": special_tokens,
            "statistics.json": statistics,
            "manifest.json": manifest,
        }

        checksums: Dict[str, str] = {}

        # 1. Serialize JSON files
        for filename, data in files_to_write.items():
            file_path = path / filename
            content = json.dumps(data, indent=2, ensure_ascii=False)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            checksums[filename] = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # 2. Serialize Markdown report
        report_path = path / "report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        checksums["report.md"] = hashlib.sha256(report_md.encode("utf-8")).hexdigest()

        # 3. Write SHA-256 checksum file
        checksum_path = path / "checksum.sha256"
        with open(checksum_path, "w", encoding="utf-8") as f:
            for filename, sha in checksums.items():
                f.write(f"{sha}  {filename}\n")

    @staticmethod
    def _parse_checksums(checksum_path: Path) -> Dict[str, str]:
        expected_checksums: Dict[str, str] = {}
        with open(checksum_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split("  ", 1)
                if len(parts) == 2:
                    sha, filename = parts
                    expected_checksums[filename] = sha
        return expected_checksums

    @staticmethod
    def _verify_file_checksum(file_path: Path, expected_sha: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        current_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if current_sha != expected_sha:
            raise SerializationError(
                f"Integrity check failed: Checksum mismatch for '{file_path.name}'."
            )
        return content

    @staticmethod
    def load(vocab_dir: str) -> Dict[str, Any]:
        """
        Loads and verifies the vocabulary files.
        """
        path = Path(vocab_dir)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary directory not found: {vocab_dir}")

        checksum_path = path / "checksum.sha256"
        if not checksum_path.exists():
            raise SerializationError(
                "Missing 'checksum.sha256' file. "
                "Vocabulary integrity cannot be verified."
            )

        # Parse expected checksum values
        expected_checksums = VocabularySerializer._parse_checksums(checksum_path)

        # Load and verify JSON files
        loaded_data: Dict[str, Any] = {}
        files_to_load = [
            "vocab.json",
            "metadata.json",
            "special_tokens.json",
            "statistics.json",
            "manifest.json",
        ]

        for filename in files_to_load:
            file_path = path / filename
            if not file_path.exists():
                raise SerializationError(
                    f"Missing expected vocabulary file: {filename}"
                )

            expected_sha = expected_checksums.get(filename, "")
            content = VocabularySerializer._verify_file_checksum(
                file_path, expected_sha
            )

            # Parse content
            key = filename.replace(".json", "")
            try:
                loaded_data[key] = json.loads(content)
            except json.JSONDecodeError as e:
                raise SerializationError(
                    f"Failed to parse JSON file '{filename}': {e}"
                ) from e

        # Optionally load report.md and check integrity
        report_path = path / "report.md"
        if report_path.exists():
            expected_sha = expected_checksums.get("report.md", "")
            content = VocabularySerializer._verify_file_checksum(
                report_path, expected_sha
            )
            loaded_data["report_md"] = content

        return loaded_data
