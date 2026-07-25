import logging
from typing import List, Optional

from datasets.registry.base_loader import BaseTextLoader
from engine.tokenizer.normalizer.unicode import UnicodeNormalizer
from engine.tokenizer.pretokenizer.split import RegexPretokenizer
from engine.tokenizer.vocabulary.cache import VocabularyCache
from engine.tokenizer.vocabulary.lifecycle import VocabularyLifecycle
from engine.tokenizer.vocabulary.manager import VocabularyManager

logger = logging.getLogger("vocabulary")


class VocabularyBuilder:
    """
    Builds a frozen, validated VocabularyManager from text dataset loader streams.
    """

    def __init__(
        self,
        normalizer: UnicodeNormalizer,
        pretokenizer: RegexPretokenizer,
        min_frequency: int = 1,
        max_vocab_size: Optional[int] = None,
        special_tokens: Optional[List[str]] = None,
        cache_dir: str = "datasets/cache",
    ) -> None:
        self.normalizer = normalizer
        self.pretokenizer = pretokenizer
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size
        self.special_tokens = special_tokens
        self.cache = VocabularyCache(cache_dir)

    def build(self, corpus_path: str, loader: BaseTextLoader) -> VocabularyManager:
        """
        Builds, validates, and locks a vocabulary from the corpus.
        """
        logger.info(f"Vocabulary build process started for dataset: {corpus_path}")

        # 1. Fetch metadata and check cache
        meta = loader.get_provenance_metadata(corpus_path)
        dataset_hash = meta["hash"]

        logger.debug(
            "Checking vocabulary frequency cache...",
            extra={"dataset_hash": dataset_hash},
        )
        frequencies = self.cache.get_frequencies(dataset_hash)

        if frequencies is not None:
            logger.info(
                "Vocabulary cache HIT. Loaded pre-computed frequencies.",
                extra={"dataset_hash": dataset_hash},
            )
        else:
            logger.info(
                "Vocabulary cache MISS. Scanning dataset file...",
                extra={"dataset_hash": dataset_hash},
            )
            frequencies = {}

            # Stream lines and accumulate frequencies
            for line in loader.load_lines(corpus_path):
                normalized = self.normalizer.normalize(line)
                tokens = self.pretokenizer.pretokenize(normalized)
                for token in tokens:
                    frequencies[token] = frequencies.get(token, 0) + 1

            # Cache the computed frequencies
            self.cache.save_frequencies(dataset_hash, frequencies)
            logger.debug(
                "Frequencies computed and cached successfully.",
                extra={"dataset_hash": dataset_hash},
            )

        # 2. Filter out tokens below min_frequency
        filtered_freqs = {
            tok: freq for tok, freq in frequencies.items() if freq >= self.min_frequency
        }
        logger.debug(
            f"Filtered tokens (min_frequency={self.min_frequency}). "
            f"Retained {len(filtered_freqs)} / {len(frequencies)} unique tokens."
        )

        # 3. Sort tokens: (Frequency descending, then lexicographically ascending)
        sorted_tokens = sorted(
            filtered_freqs.keys(), key=lambda tok: (-filtered_freqs[tok], tok)
        )

        # 4. Limit to max_vocab_size if defined
        if self.max_vocab_size is not None:
            specials_count = len(self.special_tokens) if self.special_tokens else 5
            adjusted_limit = max(0, self.max_vocab_size - specials_count)
            sorted_tokens = sorted_tokens[:adjusted_limit]
            logger.debug(
                f"Capped token list to max size {adjusted_limit} (excluding specials)."
            )

        # 5. Populate the VocabularyManager
        manager = VocabularyManager(special_tokens=self.special_tokens)
        manager.transition_to(VocabularyLifecycle.TRAINING)

        for token in sorted_tokens:
            manager.add_token(token)
            # Record frequency inside the manager's records
            manager._token_frequencies[token] = filtered_freqs[token]

        # 6. Freeze the manager (transitions to VALIDATED)
        manager.freeze()
        logger.info(
            f"Vocabulary build completed successfully. "
            f"Total vocabulary size: {len(manager)} tokens."
        )
        return manager
