import time
from typing import Any, Dict, List

from engine.tokenizer.base.tokenizer import BaseTokenizer


class TokenizerBenchmark:
    """
    Benchmarks encode and decode speeds.
    """

    @staticmethod
    def run(
        tokenizer: BaseTokenizer, samples: List[str], iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmarks token processing throughput (tokens/sec).
        """
        # 1. Benchmark Encode
        total_tokens = 0
        encoded_lists = []

        start_enc = time.perf_counter()
        for _ in range(iterations):
            for sample in samples:
                # We default add_special_tokens=False to measure raw speed
                ids = tokenizer.encode(sample, add_special_tokens=False)
                encoded_lists.append(ids)
                total_tokens += len(ids)
        end_enc = time.perf_counter()

        elapsed_enc = end_enc - start_enc

        # 2. Benchmark Decode
        start_dec = time.perf_counter()
        for _ in range(iterations):
            for ids in encoded_lists[: len(samples)]:
                _ = tokenizer.decode(ids)
        end_dec = time.perf_counter()

        # Scale decoding counts to match total iterations
        elapsed_dec = (end_dec - start_dec) * iterations

        enc_throughput = (total_tokens / elapsed_enc) if elapsed_enc > 0 else 0.0
        dec_throughput = (total_tokens / elapsed_dec) if elapsed_dec > 0 else 0.0

        return {
            "iterations": iterations,
            "total_tokens_processed": total_tokens,
            "encode_time_seconds": round(elapsed_enc, 4),
            "encode_throughput_tokens_per_sec": round(enc_throughput, 2),
            "decode_time_seconds": round(elapsed_dec, 4),
            "decode_throughput_tokens_per_sec": round(dec_throughput, 2),
        }
