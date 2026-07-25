from engine.dataset.base.dataset import TokenizedSample
from engine.dataset.packing.packer import SequencePacker
from engine.dataset.packing.dynamic import DynamicPacker
from engine.dataset.packing.truncation import TruncationHandler
from engine.dataset.packing.padding import PaddingHandler
from engine.dataset.packing.metrics import PackingMetricsCalculator


def test_sequence_packer() -> None:
    # 2 is EOS token
    samples = [
        TokenizedSample(input_ids=[1, 2, 3]),
        TokenizedSample(input_ids=[4, 5]),
        TokenizedSample(input_ids=[6, 7, 8, 9]),
    ]
    # Total tokens + EOS:
    # Sample 1: [1, 2, 3, 2] (len 4)
    # Sample 2: [4, 5, 2] (len 3)
    # Sample 3: [6, 7, 8, 9, 2] (len 5)
    # Total tokens: 12

    packer = SequencePacker(max_sequence_length=6, add_eos=True, eos_token_id=2)
    packed = list(packer.pack(iter(samples)))

    assert len(packed) == 2
    assert len(packed[0].input_ids) == 6
    assert packed[0].input_ids == [1, 2, 3, 2, 4, 5]
    assert packed[1].input_ids == [2, 6, 7, 8, 9, 2]


def test_dynamic_packer() -> None:
    samples = [
        TokenizedSample(input_ids=[1, 2]),
        TokenizedSample(input_ids=[3, 4]),
    ]
    # Sample 1: [1, 2, 2] (len 3)
    # Sample 2: [3, 4, 2] (len 3)

    packer = DynamicPacker(max_sequence_length=5, add_eos=True, eos_token_id=2)
    packed = list(packer.pack(iter(samples)))

    # Since 3 + 3 = 6 > 5, it should yield them as 2 separate sequences of length 3
    assert len(packed) == 2
    assert packed[0].input_ids == [1, 2, 2]
    assert packed[1].input_ids == [3, 4, 2]


def test_truncation_handler() -> None:
    handler = TruncationHandler(max_length=4)
    sample = TokenizedSample(input_ids=[1, 2, 3, 4, 5, 6], attention_mask=[1] * 6)
    truncated = handler.truncate(sample)

    assert len(truncated.input_ids) == 4
    assert truncated.input_ids == [1, 2, 3, 4]
    assert len(truncated.attention_mask) == 4


def test_padding_handler() -> None:
    handler = PaddingHandler(max_length=6, pad_token_id=0, ignore_index=-100)
    sample = TokenizedSample(input_ids=[1, 2, 3])
    padded = handler.pad(sample)

    assert len(padded.input_ids) == 6
    assert padded.input_ids == [1, 2, 3, 0, 0, 0]
    assert padded.attention_mask == [1, 1, 1, 0, 0, 0]
    assert padded.labels == [1, 2, 3, -100, -100, -100]


def test_packing_metrics_calculator() -> None:
    packed = [
        TokenizedSample(input_ids=[1, 2, 3, 0, 0], labels=[1, 2, 3, -100, -100])
    ]
    metrics = PackingMetricsCalculator.calculate_efficiency(
        packed, raw_lengths=[3], max_sequence_length=5, ignore_index=-100
    )

    assert metrics["valid_tokens"] == 3
    assert metrics["wasted_padding"] == 2
    assert metrics["padding_wasted_pct"] == 40.0
    assert metrics["packing_efficiency"] == 60.0
