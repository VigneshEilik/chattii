from typing import List, Tuple

import torch

from engine.dataset.base.dataset import TokenizedSample


class DataCollator:
    """Collates a list of TokenizedSamples into aligned input and target tensors."""

    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def collate(self, samples: List[TokenizedSample]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert lists to PyTorch Tensors
        input_ids_tensors = [torch.tensor(s.input_ids, dtype=torch.long) for s in samples]
        labels_tensors = [
            torch.tensor(s.labels, dtype=torch.long)
            if s.labels is not None
            else torch.tensor(s.input_ids, dtype=torch.long)
            for s in samples
        ]

        # Stack into [batch_size, sequence_length] tensors
        inputs = torch.stack(input_ids_tensors, dim=0)
        targets = torch.stack(labels_tensors, dim=0)

        return inputs, targets
