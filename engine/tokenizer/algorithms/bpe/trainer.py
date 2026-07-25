import logging
from typing import Any, Dict, List, Optional, Tuple

from datasets.registry.base_loader import BaseTextLoader
from engine.tokenizer.algorithms.bpe.checkpoint import BPECheckpointer
from engine.tokenizer.algorithms.bpe.merge_engine import BPEMergeEngine
from engine.tokenizer.algorithms.bpe.merge_history import BPEMergeHistory
from engine.tokenizer.algorithms.bpe.merge_selector import BPEMergeSelector
from engine.tokenizer.algorithms.bpe.pair_counter import BPETokenPairCounter
from engine.tokenizer.algorithms.bpe.statistics import BPEStatistics
from engine.tokenizer.algorithms.bpe.vocabulary_updater import BPEVocabularyUpdater
from engine.tokenizer.vocabulary.lifecycle import VocabularyLifecycle
from engine.tokenizer.vocabulary.manager import VocabularyManager

logger = logging.getLogger("bpe_trainer")


class BPEVocabularyTrainer:
    """
    Ingests datasets and trains Byte Pair Encoding vocabs and merge rules.
    """

    def __init__(
        self,
        normalizer: Any,
        pretokenizer: Any,
        vocab_size: int,
        special_tokens: List[str],
        checkpoint_interval: int = 100,
        output_dir: Optional[str] = None,
    ) -> None:
        self.normalizer = normalizer
        self.pretokenizer = pretokenizer
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = output_dir

    def train(
        self, corpus_path: str, loader: BaseTextLoader
    ) -> Tuple[
        VocabularyManager, List[Tuple[str, str]], BPEMergeHistory, Dict[str, Any]
    ]:
        """
        Orchestrates full iterative BPE merge training.
        """
        logger.info(f"Starting BPE training on corpus: {corpus_path}")

        # 1. Ingest and tokenize raw sentences into character lists
        word_freqs: Dict[Tuple[str, ...], int] = {}
        initial_corpus_bytes = 0

        for line in loader.load_lines(corpus_path):
            normalized = self.normalizer.normalize(line)
            tokens = self.pretokenizer.pretokenize(normalized)
            for token in tokens:
                initial_corpus_bytes += len(token.encode("utf-8"))
                char_tuple = tuple(token)
                word_freqs[char_tuple] = word_freqs.get(char_tuple, 0) + 1

        # 2. Setup initial vocabulary from unique base characters
        unique_chars = sorted(
            list(set(char for seq in word_freqs.keys() for char in seq))
        )

        manager = VocabularyManager(special_tokens=self.special_tokens)
        manager.transition_to(VocabularyLifecycle.TRAINING)

        for char in unique_chars:
            manager.add_token(char)
            # Accumulate character occurrences
            manager._token_frequencies[char] = sum(
                count for seq, count in word_freqs.items() if char in seq
            )

        logger.info(f"Initialized vocabulary with {len(unique_chars)} base characters.")

        merges: List[Tuple[str, str]] = []
        history = BPEMergeHistory()

        # 3. Iterative merging loop
        rank = 1
        target_merges = self.vocab_size - len(manager)

        logger.info(
            f"Targeting {target_merges} merge operations "
            f"to reach target vocab size {self.vocab_size}."
        )

        for _ in range(target_merges):
            pair_counts = BPETokenPairCounter.count_pairs(word_freqs)
            best_pair = BPEMergeSelector.select_best_pair(pair_counts)

            if best_pair is None:
                logger.info(
                    "No candidate adjacent pairs found to merge. Terminating early."
                )
                break

            pair_freq = pair_counts[best_pair]
            merged_token = best_pair[0] + best_pair[1]

            # Update corpus
            word_freqs = BPEMergeEngine.merge_corpus(word_freqs, best_pair)

            # Update vocab manager
            BPEVocabularyUpdater.update_vocabulary(manager, merged_token, pair_freq)

            # Record history
            merges.append(best_pair)
            history.record(rank, best_pair, merged_token, pair_freq)

            logger.debug(
                f"Merge #{rank}: {best_pair} -> '{merged_token}' "
                f"(frequency: {pair_freq})"
            )

            # Calculate metrics
            current_tokens_count = sum(
                len(seq) * count for seq, count in word_freqs.items()
            )

            # Periodic checkpoints
            if self.output_dir and rank % self.checkpoint_interval == 0:
                stats = BPEStatistics.compute(
                    vocab_size=len(manager),
                    num_merges=rank,
                    initial_corpus_bytes=initial_corpus_bytes,
                    current_corpus_tokens_count=current_tokens_count,
                    vocab_map=manager._token_to_id,
                )
                BPECheckpointer.save_checkpoint(
                    output_dir=self.output_dir,
                    merge_count=rank,
                    vocab_map=manager._token_to_id,
                    merges=merges,
                    statistics=stats,
                )

            rank += 1
            if len(manager) >= self.vocab_size:
                break

        # 4. Finalize training
        final_tokens_count = sum(len(seq) * count for seq, count in word_freqs.items())
        final_stats = BPEStatistics.compute(
            vocab_size=len(manager),
            num_merges=len(merges),
            initial_corpus_bytes=initial_corpus_bytes,
            current_corpus_tokens_count=final_tokens_count,
            vocab_map=manager._token_to_id,
        )

        manager.freeze()
        logger.info(f"BPE training complete. Final vocab size: {len(manager)}")
        return manager, merges, history, final_stats
