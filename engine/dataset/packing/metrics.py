from typing import List

from engine.dataset.base.dataset import TokenizedSample


class PackingMetricsCalculator:
    """Computes context utilization, padding waste percentage, and overall packing efficiency."""

    @staticmethod
    def calculate_efficiency(
        packed_samples: List[TokenizedSample],
        raw_lengths: List[int],
        max_sequence_length: int,
        ignore_index: int = -100,
    ) -> dict:
        total_raw_tokens = sum(raw_lengths)
        total_packed_tokens = sum(len(s.input_ids) for s in packed_samples)

        # Count actual content tokens (excluding ignore_index padding values)
        valid_tokens = 0
        for sample in packed_samples:
            if sample.labels is not None:
                valid_tokens += sum(1 for label in sample.labels if label != ignore_index)
            else:
                valid_tokens += len(sample.input_ids)

        wasted_padding = total_packed_tokens - valid_tokens
        padding_wasted_pct = (wasted_padding / max(1, total_packed_tokens)) * 100
        packing_efficiency = (valid_tokens / max(1, total_packed_tokens)) * 100

        return {
            "total_raw_tokens": total_raw_tokens,
            "total_packed_slots": total_packed_tokens,
            "valid_tokens": valid_tokens,
            "wasted_padding": wasted_padding,
            "padding_wasted_pct": padding_wasted_pct,
            "packing_efficiency": packing_efficiency,
            "context_utilization": packing_efficiency,
        }
