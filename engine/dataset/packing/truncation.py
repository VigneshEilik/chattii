from engine.dataset.base.dataset import TokenizedSample


class TruncationHandler:
    """Truncates token sequences to a fixed maximum length if sequence packing is disabled."""

    def __init__(self, max_length: int = 2048) -> None:
        self.max_length = max_length

    def truncate(self, sample: TokenizedSample) -> TokenizedSample:
        ids = sample.input_ids
        mask = sample.attention_mask
        labels = sample.labels

        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
            if mask is not None:
                mask = mask[:self.max_length]
            if labels is not None:
                labels = labels[:self.max_length]

        return TokenizedSample(
            input_ids=ids,
            sample_id=sample.sample_id,
            attention_mask=mask,
            labels=labels,
            metadata=sample.metadata.copy(),
        )
