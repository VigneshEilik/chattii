"""
Chatii Milestone 5 -- Dataset Pipeline Demo
============================================
Demonstrates the full dataset engine: register, inspect, validate, stream,
benchmark, and report generation.

Usage:
    python scripts/dataset_demo.py
"""

import os
import sys
import tempfile

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.dataset.base.dataset import TextSample, TokenizedSample  # noqa: E402
from engine.dataset.binary.mmap import MemoryMappedDataset  # noqa: E402
from engine.dataset.binary.reader import BinaryDatasetReader  # noqa: E402
from engine.dataset.binary.writer import BinaryDatasetWriter  # noqa: E402
from engine.dataset.cache.fingerprint import DatasetFingerprint  # noqa: E402
from engine.dataset.cleaner.html import HTMLCleaner  # noqa: E402
from engine.dataset.cleaner.pipeline import CleaningPipeline  # noqa: E402
from engine.dataset.cleaner.unicode import UnicodeCleaner  # noqa: E402
from engine.dataset.cleaner.whitespace import WhitespaceCleaner  # noqa: E402
from engine.dataset.packing.metrics import PackingMetricsCalculator  # noqa: E402
from engine.dataset.packing.packer import SequencePacker  # noqa: E402
from engine.dataset.packing.padding import PaddingHandler  # noqa: E402
from engine.dataset.packing.truncation import TruncationHandler  # noqa: E402
from engine.dataset.reports.report_generator import DatasetReportGenerator  # noqa: E402
from engine.dataset.service import DatasetService  # noqa: E402
from engine.dataset.statistics.calculator import (
    DatasetStatisticsCalculator,  # noqa: E402
)
from engine.dataset.validator.quality import QualityValidator  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("  Chatii -- Milestone 5: Dataset Pipeline Demo")
    print("=" * 60)

    # -- 1. Create sample corpus --
    corpus_dir = os.path.join("datasets", "demo")
    os.makedirs(corpus_dir, exist_ok=True)
    corpus_file = os.path.join(corpus_dir, "sample_corpus.txt")

    lines = [
        "Chatii is a production-grade modular transformer framework.",
        "   <b>HTML  noise</b>  and  extra   whitespace   here.  ",
        "Short",
        "Transformers revolutionized natural language processing in 2017.",
        "Large language models like GPT, LLaMA, and Gemma are trained on massive datasets.",
        "Tokenization converts raw text into integer token IDs for embedding lookups.",
        "Sequence packing concatenates multiple samples to minimise padding waste.",
        "Memory-mapped binary shards enable O(1) random access during training.",
        "Mixed-precision training reduces GPU memory and improves throughput.",
        "Gradient accumulation simulates larger effective batch sizes on limited hardware.",
    ]
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[OK] Sample corpus written to: {corpus_file}")

    # -- 2. Cleaning Pipeline --
    print("\n--- Step 2: Cleaning Pipeline ---")
    pipeline = CleaningPipeline(
        cleaners=[HTMLCleaner(), UnicodeCleaner(), WhitespaceCleaner()]
    )
    raw = TextSample(text="  <p>Hello &amp;  World</p>  ")
    cleaned = pipeline.clean(raw)
    print(f"  Raw:     '{raw.text}'")
    print(f"  Cleaned: '{cleaned.text}'")

    # -- 3. Validation --
    print("\n--- Step 3: Quality Validation ---")
    qv = QualityValidator(min_length=10, max_length=1000)
    for text in ["Short", "This sentence is long enough to pass."]:
        valid, reason = qv.validate(TextSample(text))
        status = "PASS" if valid else f"FAIL ({reason})"
        print(f"  '{text[:40]}' -> {status}")

    # -- 4. Sequence Packing --
    print("\n--- Step 4: Sequence Packing ---")
    tokenized = [
        TokenizedSample(input_ids=[1, 2, 3]),
        TokenizedSample(input_ids=[4, 5]),
        TokenizedSample(input_ids=[6, 7, 8, 9]),
    ]
    packer = SequencePacker(max_sequence_length=6, add_eos=True, eos_token_id=2)
    packed = list(packer.pack(iter(tokenized)))
    for i, seq in enumerate(packed):
        print(f"  Packed sequence {i}: {seq.input_ids}")

    # -- 5. Padding & Truncation --
    print("\n--- Step 5: Padding & Truncation ---")
    padder = PaddingHandler(max_length=8, pad_token_id=0, ignore_index=-100)
    truncator = TruncationHandler(max_length=4)
    sample = TokenizedSample(input_ids=[10, 20, 30])
    padded = padder.pad(sample)
    print(f"  Padded:    {padded.input_ids}")
    print(f"  Attn mask: {padded.attention_mask}")
    long_sample = TokenizedSample(input_ids=[1, 2, 3, 4, 5, 6, 7])
    truncated = truncator.truncate(long_sample)
    print(f"  Truncated: {truncated.input_ids}")

    # -- 6. Packing Efficiency Metrics --
    print("\n--- Step 6: Packing Efficiency ---")
    metrics = PackingMetricsCalculator.calculate_efficiency(
        packed_samples=[padded],
        raw_lengths=[3],
        max_sequence_length=8,
        ignore_index=-100,
    )
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # -- 7. Binary Writer / Mmap Reader --
    print("\n--- Step 7: Binary Shard Writer & Mmap Reader ---")
    with tempfile.TemporaryDirectory() as tmp:
        writer = BinaryDatasetWriter(out_dir=tmp, prefix="demo", dtype="uint16")
        samples_to_write = [
            TokenizedSample(input_ids=[100, 200, 300, 400]),
            TokenizedSample(input_ids=[500, 600, 700, 800]),
        ]
        shard_paths = writer.write(iter(samples_to_write), tokens_per_shard=4)
        print(f"  Written {len(shard_paths)} shard(s)")

        mmap_ds = MemoryMappedDataset(shard_paths, dtype="uint16")
        print(f"  Total tokens in mmap: {len(mmap_ds)}")
        print(f"  Token[0] = {mmap_ds[0]},  Token[4] = {mmap_ds[4]}")

        reader = BinaryDatasetReader(mmap_ds, block_size=4)
        for chunk in reader.stream():
            print(f"  Block: {chunk.input_ids}")
        mmap_ds.close()

    # -- 8. Statistics & Report --
    print("\n--- Step 8: Dataset Statistics & Report ---")
    calc = DatasetStatisticsCalculator()
    demo_samples = [
        (TextSample("Hello world"), TokenizedSample(input_ids=[10, 20, 2])),
        (
            TextSample("Large language model training"),
            TokenizedSample(input_ids=[5, 6, 7, 8, 2]),
        ),
    ]
    for ts, tok in demo_samples:
        calc.add_sample(ts, tok)
    stats = calc.calculate(vocab_size=100)
    print(f"  Samples: {stats['sample_count']}")
    print(f"  Tokens:  {stats['token_count']}")
    print(f"  Avg chars/sample: {stats['char_stats']['average']:.1f}")

    report_path = os.path.join(corpus_dir, "dataset_report.md")
    DatasetReportGenerator.generate_report(stats, report_path)
    print(f"  Report saved to: {report_path}")

    # -- 9. Cache Fingerprinting --
    print("\n--- Step 9: Cache Fingerprinting ---")
    fp = DatasetFingerprint.calculate(
        config_dict={"batch_size": 16, "seq_len": 512},
        file_paths=[corpus_file],
    )
    print(f"  Fingerprint: {fp}")

    # -- 10. Dataset Service (Registry) --
    print("\n--- Step 10: Dataset Service ---")
    service = DatasetService()
    service.register_dataset("demo_corpus", corpus_file, fmt="txt")
    service.list_datasets()

    print("\n" + "=" * 60)
    print("  [OK] Milestone 5 Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
