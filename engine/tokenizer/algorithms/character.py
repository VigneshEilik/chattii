import json
import os
from typing import Dict, List

from engine.tokenizer.base.tokenizer import BaseTokenizer
from engine.tokenizer.normalizer.unicode import UnicodeNormalizer
from engine.tokenizer.pretokenizer.split import RegexPretokenizer
from engine.tokenizer.vocabulary.manager import VocabularyManager


class CharacterTokenizer(BaseTokenizer):
    """
    Tokenizer that maps individual characters to unique vocabulary IDs.
    """

    def __init__(
        self, unicode_normalization: str = "NFKC", lowercase: bool = False
    ) -> None:
        self.vocab_manager = VocabularyManager()
        self.normalizer = UnicodeNormalizer(unicode_normalization, lowercase)
        self.pretokenizer = RegexPretokenizer(mode="char")

    def train(self, corpus_path: str) -> None:
        """
        Scans a text corpus for unique characters and adds them to the vocabulary.
        """
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Training corpus not found: {corpus_path}")

        unique_chars = set()
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                normalized = self.normalizer.normalize(line)
                pretokenized = self.pretokenizer.pretokenize(normalized)
                unique_chars.update(pretokenized)

        # Register unique characters in sorted order for determinism
        for char in sorted(list(unique_chars)):
            self.vocab_manager.add_token(char)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Normalizes and splits text into character IDs.
        """
        normalized = self.normalizer.normalize(text)
        chars = self.pretokenizer.pretokenize(normalized)
        ids = [self.vocab_manager.token_to_id(char) for char in chars]

        if add_special_tokens:
            specials = self.vocab_manager.get_special_tokens()
            if "bos" in specials:
                ids.insert(0, self.vocab_manager.token_to_id(specials["bos"]))
            if "eos" in specials:
                ids.append(self.vocab_manager.token_to_id(specials["eos"]))

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Converts character IDs back into a reconstructed text string.
        """
        tokens = []
        specials = set(self.vocab_manager.get_special_tokens().values())
        for token_id in ids:
            token = self.vocab_manager.id_to_token(token_id)
            if token not in specials:
                tokens.append(token)
        return "".join(tokens)

    def save(self, path: str) -> None:
        """
        Saves the tokenizer metadata and vocabulary to path directory.
        """
        os.makedirs(path, exist_ok=True)
        vocab_data = {
            "version": "1.0.0",
            "algorithm": "character",
            "vocab": self.vocab_manager._token_to_id,
            "special_tokens": self.vocab_manager.get_special_tokens(),
        }
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)

    def load(self, path: str) -> None:
        """
        Loads the tokenizer metadata and vocabulary from path directory.
        """
        vocab_file = os.path.join(path, "tokenizer.json")
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Tokenizer configuration not found: {vocab_file}")

        with open(vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_manager = VocabularyManager(list(data["special_tokens"].values()))
        self.vocab_manager._token_to_id = data["vocab"]
        self.vocab_manager._id_to_token = {v: k for k, v in data["vocab"].items()}

    def get_vocab_size(self) -> int:
        return len(self.vocab_manager)

    def get_special_tokens(self) -> Dict[str, str]:
        return self.vocab_manager.get_special_tokens()
