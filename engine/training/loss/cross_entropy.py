# ruff: noqa: E501
import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """
    Custom Cross Entropy loss implementation supporting label smoothing and ignore index.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes cross-entropy loss over logits and target distributions.
        """
        # Flatten batch and sequence dimensions if input is 3D (B, T, C)
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

        # Create mask to exclude padding/ignore index targets
        mask = targets != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        active_logits = logits[mask]
        active_targets = targets[mask]

        # Numerically stable Log-Softmax implementation
        logits_max = torch.max(active_logits, dim=-1, keepdim=True).values
        logits_stable = active_logits - logits_max
        log_sum_exp = torch.log(
            torch.sum(torch.exp(logits_stable), dim=-1, keepdim=True)
        )
        log_probs = logits_stable - log_sum_exp

        # Extract log probability of target index class
        target_log_probs = log_probs.gather(
            dim=-1, index=active_targets.unsqueeze(-1)
        ).squeeze(-1)

        # Calculate loss incorporating label smoothing if requested
        if self.label_smoothing > 0.0:
            vocab_size = logits.size(-1)
            sum_log_probs = log_probs.sum(dim=-1)
            loss = (
                -(1.0 - self.label_smoothing) * target_log_probs
                - (self.label_smoothing / vocab_size) * sum_log_probs
            )
        else:
            loss = -target_log_probs

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
