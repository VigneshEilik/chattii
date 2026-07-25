# ruff: noqa: C901
import json
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import Dataset

from engine.tokenizer.service import TokenizerService
from engine.training.dataset.packing import SequencePacker
from engine.training.preprocessing import (
    DocumentDeduplicator,
    LanguageDetector,
    QualityFilter,
    TextCleaner,
)


class ChatiiDataset(Dataset):
    """
    Production-grade Dataset loader that filters, tokenizes, and packs documents.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer: TokenizerService,
        sequence_length: int,
        eos_token_id: int = 0,
        clean: bool = True,
        deduplicate: bool = True,
        language_filter: Optional[str] = "en",
        quality_filter: bool = True,
    ) -> None:
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.eos_token_id = eos_token_id

        # 1. Load documents from file
        raw_docs: List[str] = []
        if self.file_path.suffix == ".jsonl":
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if "text" in item:
                                raw_docs.append(item["text"])
                        except Exception:
                            pass
        else:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Split into documents on double newline
                raw_docs = [p.strip() for p in content.split("\n\n") if p.strip()]

        # 2. Preprocess documents
        dedup = DocumentDeduplicator()
        cleaned_docs: List[str] = []
        for doc in raw_docs:
            if clean:
                doc = TextCleaner.clean(doc)
            if deduplicate and dedup.is_duplicate(doc):
                continue
            if language_filter and not LanguageDetector.is_target_language(
                doc, target=language_filter
            ):
                continue
            if quality_filter and not QualityFilter.is_high_quality(doc):
                continue
            cleaned_docs.append(doc)

        # 3. Tokenize documents
        tokenized_docs: List[List[int]] = []
        for doc in cleaned_docs:
            tokens = self.tokenizer.encode(doc)
            tokenized_docs.append(tokens)

        # 4. Dynamic Packing (Yields size: sequence_length + 1)
        self.packed_blocks = list(
            SequencePacker.pack(
                tokenized_docs,
                sequence_length=self.sequence_length,
                eos_token_id=self.eos_token_id,
            )
        )

    def __len__(self) -> int:
        return len(self.packed_blocks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.packed_blocks[idx], dtype=torch.long)
