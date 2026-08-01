from engine.dataset.base.dataset import TokenizedSample


class PaddingHandler:
    """Pads short token sequences to the target context length using padding tokens."""

    def __init__(
        self, max_length: int = 2048, pad_token_id: int = 0, ignore_index: int = -100
    ) -> None:
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def pad(self, sample: TokenizedSample) -> TokenizedSample:
        ids = sample.input_ids
        length = len(ids)

        if length >= self.max_length:
            return sample

        pad_len = self.max_length - length
        padded_ids = ids + [self.pad_token_id] * pad_len

        # Construct padded attention mask (0 on padded tokens)
        mask = sample.attention_mask or [1] * length
        padded_mask = mask + [0] * pad_len

        # Construct padded labels (ignore index on padded labels so loss isn't computed for them)
        labels = sample.labels or ids.copy()
        padded_labels = labels + [self.ignore_index] * pad_len

        return TokenizedSample(
            input_ids=padded_ids,
            sample_id=sample.sample_id,
            attention_mask=padded_mask,
            labels=padded_labels,
            metadata=sample.metadata.copy(),
        )
