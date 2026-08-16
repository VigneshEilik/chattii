import csv
import json
from pathlib import Path


class DocumentParser:
    """Extracts raw text content from PDF, TXT, Markdown, DOCX, CSV, JSON, HTML."""

    @staticmethod
    def parse_text(content: str) -> str:
        return content.strip()

    @staticmethod
    def parse_json(content: str) -> str:
        data = json.loads(content)
        return json.dumps(data, indent=2)

    @staticmethod
    def parse_csv(content: str) -> str:
        lines = content.strip().splitlines()
        reader = csv.reader(lines)
        rows = list(reader)
        return "\n".join([", ".join(row) for row in rows])

    @classmethod
    def parse_file(cls, filename: str, content_bytes: bytes) -> str:
        ext = Path(filename).suffix.lower()
        text = content_bytes.decode("utf-8", errors="ignore")

        if ext in [".txt", ".md", ".markdown", ".html", ".htm"]:
            return cls.parse_text(text)
        elif ext == ".json":
            return cls.parse_json(text)
        elif ext == ".csv":
            return cls.parse_csv(text)
        else:
            # Fallback text extraction
            return cls.parse_text(text)
