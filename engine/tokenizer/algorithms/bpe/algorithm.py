import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.tokenizer.algorithms.bpe.decoder import BPEDecoder
from engine.tokenizer.algorithms.bpe.encoder import BPEEncoder
from engine.tokenizer.algorithms.bpe.exceptions import BPEError
from engine.tokenizer.algorithms.bpe.merge_history import BPEMergeHistory
from engine.tokenizer.algorithms.bpe.trainer import BPEVocabularyTrainer
from engine.tokenizer.base.tokenizer import BaseTokenizer
from engine.tokenizer.normalizer.unicode import UnicodeNormalizer
from engine.tokenizer.pretokenizer.split import RegexPretokenizer
from engine.tokenizer.serialization.io import VocabularySerializer
from engine.tokenizer.vocabulary.lifecycle import VocabularyLifecycle
from engine.tokenizer.vocabulary.manager import VocabularyManager
from engine.tokenizer.vocabulary.manifest import VocabularyManifest
from engine.tokenizer.vocabulary.metadata import VocabularyMetadata


class BPETokenizer(BaseTokenizer):
    """
    Concrete Byte Pair Encoding (BPE) Tokenizer implementation.
    """

    def __init__(
        self,
        unicode_normalization: str = "NFKC",
        lowercase: bool = False,
        min_frequency: int = 1,
        max_vocab_size: int = 1000,
        special_tokens: Optional[List[str]] = None,
        checkpoint_interval: int = 100,
    ) -> None:
        self.unicode_normalization = unicode_normalization
        self.lowercase = lowercase
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size
        self.special_tokens = special_tokens or [
            "<pad>",
            "<unk>",
            "<s>",
            "</s>",
            "<mask>",
        ]
        self.checkpoint_interval = checkpoint_interval

        self.normalizer = UnicodeNormalizer(unicode_normalization, lowercase)
        self.pretokenizer = RegexPretokenizer(mode="word")

        self.vocab_manager: Optional[VocabularyManager] = None
        self.merges: List[Tuple[str, str]] = []
        self.merge_history = BPEMergeHistory()
        self.statistics: Dict[str, Any] = {}

        self.encoder: Optional[BPEEncoder] = None
        self.decoder: Optional[BPEDecoder] = None

    def train(self, corpus_path: str, loader: Optional[Any] = None) -> None:
        """
        Trains BPE subwords and merge rules on a target dataset corpus.
        """
        if loader is None:
            from datasets.registry.text_loader import TextLoader

            loader = TextLoader()

        trainer = BPEVocabularyTrainer(
            normalizer=self.normalizer,
            pretokenizer=self.pretokenizer,
            vocab_size=self.max_vocab_size,
            special_tokens=self.special_tokens,
            checkpoint_interval=self.checkpoint_interval,
            output_dir=None,
        )

        manager, merges, history, stats = trainer.train(corpus_path, loader)
        self.vocab_manager = manager
        self.merges = merges
        self.merge_history = history
        self.statistics = stats

        merges_dict = {pair: idx for idx, pair in enumerate(merges)}
        self.encoder = BPEEncoder(manager, merges_dict)
        self.decoder = BPEDecoder(manager)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Splits and encodes text using greedy subword BPE rules.
        """
        if not self.encoder or not self.vocab_manager:
            raise BPEError("BPE Tokenizer has not been trained or loaded yet.")

        normalized = self.normalizer.normalize(text)
        pre_tokens = self.pretokenizer.pretokenize(normalized)

        ids = []
        if add_special_tokens:
            bos_token = self.vocab_manager.get_special_tokens().get("bos", "<s>")
            ids.append(self.vocab_manager.get_id(bos_token))

        for word in pre_tokens:
            subwords = self.encoder.encode_word(word)
            for sub in subwords:
                ids.append(self.vocab_manager.get_id(sub))

        if add_special_tokens:
            eos_token = self.vocab_manager.get_special_tokens().get("eos", "</s>")
            ids.append(self.vocab_manager.get_id(eos_token))

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Reconstructs the original string from sequence of token IDs.
        """
        if not self.decoder:
            raise BPEError("BPE Tokenizer has not been trained or loaded yet.")
        return self.decoder.decode(ids)

    def save(self, vocab_dir: str) -> None:
        """
        Serializes both standard vocabulary maps and BPE merges rule tables.
        """
        if not self.vocab_manager:
            raise BPEError("Vocabulary manager not initialized. Train or load first.")

        path = Path(vocab_dir)
        path.mkdir(parents=True, exist_ok=True)

        # 1. Create metadata and manifest configurations
        metadata = VocabularyMetadata(
            algorithm="bpe",
            vocab_size=len(self.vocab_manager),
            reserved_tokens=list(self.vocab_manager.get_special_tokens().values()),
            normalization=self.unicode_normalization,
            lowercase=self.lowercase,
            min_frequency=self.min_frequency,
            dataset_provenance=self.statistics.get(
                "provenance", {"name": "bpe_training"}
            ),
        )

        manifest = VocabularyManifest(
            name="default_bpe",
            algorithm="bpe",
            vocab_size=len(self.vocab_manager),
            location=vocab_dir,
        )

        # 2. Serialize BPE merges.txt
        merges_path = path / "merges.txt"
        merges_content = "#version: 1.0.0\n"
        for pair in self.merges:
            merges_content += f"{pair[0]} {pair[1]}\n"
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write(merges_content)

        # 3. Serialize merge_history.json
        history_path = path / "merge_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.merge_history.get_history(), f, indent=2)

        # 4. Serialize report.md
        report_path = path / "report.md"
        report_md = (
            f"# BPE Vocabulary Report\n"
            f"- **Algorithm**: Byte Pair Encoding (BPE)\n"
            f"- **Vocab Size**: {len(self.vocab_manager)}\n"
            f"- **Merges Count**: {len(self.merges)}\n"
            f"- **Average Length**: "
            f"{self.statistics.get('average_token_length', 0.0)} chars\n"
            f"- **Compression Ratio**: "
            f"{self.statistics.get('compression_ratio', 1.0)}\n"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        # 5. Serialize standard vocab formats
        VocabularySerializer.save(
            vocab_dir=vocab_dir,
            vocab_map=self.vocab_manager._token_to_id,
            metadata=metadata.model_dump(),
            special_tokens=self.vocab_manager.get_special_tokens(),
            statistics=self.statistics,
            manifest=manifest.model_dump(),
            report_md=report_md,
        )

        # 6. Recalculate checksum.sha256 to include merges.txt and merge_history.json
        files_to_hash = [
            "vocab.json",
            "metadata.json",
            "special_tokens.json",
            "statistics.json",
            "manifest.json",
            "report.md",
            "merges.txt",
            "merge_history.json",
        ]
        checksums = {}
        for fname in files_to_hash:
            fpath = path / fname
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                checksums[fname] = hashlib.sha256(content.encode("utf-8")).hexdigest()

        checksum_path = path / "checksum.sha256"
        with open(checksum_path, "w", encoding="utf-8") as f:
            for fname, sha in checksums.items():
                f.write(f"{sha}  {fname}\n")

    def _parse_merges(self, merges_path: Path) -> List[Tuple[str, str]]:
        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
        return merges

    def _parse_history(self, history_path: Path) -> BPEMergeHistory:
        with open(history_path, "r", encoding="utf-8") as f:
            history_list = json.load(f)
        history = BPEMergeHistory()
        for item in history_list:
            history.record(
                item["rank"],
                (item["pair"][0], item["pair"][1]),
                item["merged"],
                item["frequency"],
            )
        return history

    def load(self, vocab_dir: str) -> None:
        """
        Loads and verifies a saved BPE vocabulary registry directory.
        """
        path = Path(vocab_dir)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary directory not found: {vocab_dir}")

        checksum_path = path / "checksum.sha256"
        if not checksum_path.exists():
            raise BPEError(
                "Missing 'checksum.sha256' file. Integrity cannot be verified."
            )

        # Parse checksum values
        expected_checksums = {}
        with open(checksum_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split("  ", 1)
                if len(parts) == 2:
                    sha, filename = parts
                    expected_checksums[filename] = sha

        # Verify BPE specific file hashes
        bpe_files = ["merges.txt", "merge_history.json"]
        for fname in bpe_files:
            fpath = path / fname
            if not fpath.exists():
                raise BPEError(f"Missing BPE file: {fname}")
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            current_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if current_sha != expected_checksums.get(fname):
                raise BPEError(
                    f"Integrity check failed: Checksum mismatch for '{fname}'."
                )

        # Load standard files
        data = VocabularySerializer.load(vocab_dir)

        # Reconstruct VocabularyManager
        special_tokens_vals = list(data["special_tokens"].values())
        self.vocab_manager = VocabularyManager(special_tokens=special_tokens_vals)
        self.vocab_manager._token_to_id = data["vocab"]
        self.vocab_manager._id_to_token = {v: k for k, v in data["vocab"].items()}
        self.vocab_manager._lifecycle = VocabularyLifecycle.VALIDATED
        self.statistics = data["statistics"]

        self.merges = self._parse_merges(path / "merges.txt")
        self.merge_history = self._parse_history(path / "merge_history.json")

        # Initialize helper modules
        merges_dict = {pair: idx for idx, pair in enumerate(self.merges)}
        self.encoder = BPEEncoder(self.vocab_manager, merges_dict)
        self.decoder = BPEDecoder(self.vocab_manager)

    def get_special_tokens(self) -> Dict[str, str]:
        if not self.vocab_manager:
            return {
                "pad": "<pad>",
                "unk": "<unk>",
                "bos": "<s>",
                "eos": "</s>",
                "mask": "<mask>",
            }
        return self.vocab_manager.get_special_tokens()

    def get_vocab_size(self) -> int:
        if not self.vocab_manager:
            return 0
        return len(self.vocab_manager)
