# ruff: noqa: E501
import torch


class LabelSmoother:
    """Helper to convert target index classes to smoothed probability distributions."""

    @staticmethod
    def smooth(
        targets: torch.Tensor,
        vocab_size: int,
        epsilon: float = 0.1,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        Converts hard target indices (N,) to soft probability distributions (N, vocab_size).
        """
        device = targets.device
        num_samples = targets.size(0)

        # Initialize uniform target distribution mass
        smoothed = torch.full(
            (num_samples, vocab_size),
            epsilon / vocab_size,
            device=device,
            dtype=torch.float32,
        )

        # Allocate remaining target probability mass on original index classes
        targets_unsqueezed = targets.unsqueeze(-1)
        # Filter negative indices (like ignore_index=-100) to avoid scatter errors
        valid_mask = targets >= 0
        if valid_mask.any():
            valid_targets = targets_unsqueezed.clone()
            valid_targets[~valid_mask] = 0
            smoothed.scatter_(
                dim=-1,
                index=valid_targets,
                value=1.0 - epsilon + (epsilon / vocab_size),
            )

        # Nullify ignore index sequences
        invalid_mask = targets == ignore_index
        smoothed[invalid_mask] = 0.0
        return smoothed
