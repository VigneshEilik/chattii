import os
import tempfile

import numpy as np

from engine.dataset.base.dataset import TokenizedSample
from engine.dataset.binary.mmap import MemoryMappedDataset
from engine.dataset.binary.reader import BinaryDatasetReader
from engine.dataset.binary.writer import BinaryDatasetWriter


def test_binary_writer_and_reader() -> None:
    samples = [
        TokenizedSample(input_ids=[10, 20, 30, 40]),
        TokenizedSample(input_ids=[50, 60, 70, 80]),
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        writer = BinaryDatasetWriter(
            out_dir=temp_dir, prefix="test_bin", dtype="uint16"
        )
        # Split into shards of 4 tokens
        shard_paths = writer.write(iter(samples), tokens_per_shard=4)

        assert len(shard_paths) == 2
        assert os.path.exists(shard_paths[0])
        assert os.path.exists(shard_paths[1])

        # Test MemoryMappedDataset
        mmap_ds = MemoryMappedDataset(shard_paths, dtype="uint16")
        assert len(mmap_ds) == 8
        assert mmap_ds[0] == 10
        assert mmap_ds[4] == 50

        # Slice checks
        sub_arr = mmap_ds[2:6]
        assert np.array_equal(sub_arr, np.array([30, 40, 50, 60], dtype=np.uint16))

        # Test BinaryDatasetReader
        reader = BinaryDatasetReader(mmap_ds, block_size=4)
        chunks = list(reader.stream())
        assert len(chunks) == 2
        assert chunks[0].input_ids == [10, 20, 30, 40]
        assert chunks[1].input_ids == [50, 60, 70, 80]
        mmap_ds.close()
