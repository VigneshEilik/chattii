import torch
from engine.dataset.base.dataset import TokenizedSample
from engine.dataset.batching.collator import DataCollator
from engine.dataset.batching.dataloader import ChatiiDataLoader
from engine.dataset.batching.samplers import (
    SequentialSampler,
    RandomSampler,
    WeightedSampler,
    DistributedSampler,
    TemperatureSampler,
)


def test_data_collator() -> None:
    samples = [
        TokenizedSample(input_ids=[1, 2, 3], labels=[1, 2, 3]),
        TokenizedSample(input_ids=[4, 5, 6], labels=[4, 5, 6]),
    ]
    collator = DataCollator()
    inputs, targets = collator.collate(samples)

    assert isinstance(inputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert inputs.shape == (2, 3)
    assert targets.shape == (2, 3)


def test_chatii_dataloader() -> None:
    samples = [
        TokenizedSample(input_ids=[1, 2], labels=[1, 2]),
        TokenizedSample(input_ids=[3, 4], labels=[3, 4]),
        TokenizedSample(input_ids=[5, 6], labels=[5, 6]),
    ]
    # Disable prefetch for synchronous tests
    dataloader = ChatiiDataLoader(iter(samples), batch_size=2, prefetch_batches=0)
    batches = list(dataloader)

    assert len(batches) == 2
    inputs1, targets1 = batches[0]
    assert inputs1.shape == (2, 2)
    inputs2, targets2 = batches[1]
    assert inputs2.shape == (1, 2)


def test_samplers() -> None:
    data = [10, 20, 30, 40, 50]

    # 1. SequentialSampler
    seq_sampler = SequentialSampler(data)
    assert list(seq_sampler) == [0, 1, 2, 3, 4]
    assert len(seq_sampler) == 5

    # 2. RandomSampler
    rand_sampler = RandomSampler(data)
    assert len(list(rand_sampler)) == 5
    assert set(rand_sampler) == {0, 1, 2, 3, 4}

    # 3. WeightedSampler
    weighted_sampler = WeightedSampler(weights=[0.1, 0.8, 0.1, 0.0, 0.0], num_samples=10)
    indices = list(weighted_sampler)
    assert len(indices) == 10
    assert all(idx in (0, 1, 2) for idx in indices)

    # 4. DistributedSampler
    dist_sampler = DistributedSampler(data, num_replicas=2, rank=0, shuffle=False)
    assert list(dist_sampler) == [0, 2, 4]

    # 5. TemperatureSampler
    temp_sampler = TemperatureSampler(weights=[0.1, 0.8, 0.1, 0.0, 0.0], num_samples=10, temperature=0.5)
    assert len(list(temp_sampler)) == 10
