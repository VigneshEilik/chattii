import torch

from engine.inference.logits.processor import LogitsProcessor


class RepetitionPenaltyProcessor(LogitsProcessor):
    """Applies repetition scaling to penalize tokens already present in the prompt/generation history."""

    def __init__(self, penalty: float = 1.0) -> None:
        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.penalty == 1.0 or input_ids.numel() == 0:
            return scores

        # input_ids shape: (batch_size, seq_len)
        # scores shape: (batch_size, vocab_size)
        for i in range(scores.size(0)):
            unique_ids = torch.unique(input_ids[i])
            for token_id in unique_ids:
                val = scores[i, token_id].item()
                if val < 0.0:
                    scores[i, token_id] = val * self.penalty
                else:
                    scores[i, token_id] = val / self.penalty
        return scores


class FrequencyPenaltyProcessor(LogitsProcessor):
    """Applies frequency scaling to penalize tokens based on their historical occurrence frequency."""

    def __init__(self, penalty: float = 0.0) -> None:
        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.penalty == 0.0 or input_ids.numel() == 0:
            return scores

        for i in range(scores.size(0)):
            # Count occurrences of each token id
            unique, counts = torch.unique(input_ids[i], return_counts=True)
            for token_id, count in zip(unique, counts):
                scores[i, token_id] -= count.item() * self.penalty
        return scores


class PresencePenaltyProcessor(LogitsProcessor):
    """Applies presence scaling to penalize tokens based on their existence in history."""

    def __init__(self, penalty: float = 0.0) -> None:
        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.penalty == 0.0 or input_ids.numel() == 0:
            return scores

        for i in range(scores.size(0)):
            unique = torch.unique(input_ids[i])
            for token_id in unique:
                scores[i, token_id] -= self.penalty
        return scores
