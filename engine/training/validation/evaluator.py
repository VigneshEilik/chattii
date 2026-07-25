# ruff: noqa: E501
from typing import Any, Dict

import torch
import torch.nn as nn

from engine.training.validation.perplexity import PerplexityEvaluator


class SFTValEvaluator:
    """Orchestrates validation cycles over target datasets to compute validation loss/metrics."""

    @staticmethod
    def evaluate(
        model: nn.Module, dataloader: Any, loss_fn: nn.Module, device: torch.device
    ) -> Dict[str, Any]:
        """
        Runs model evaluation, computes token accuracy, and returns metrics dict.
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        total_batches = len(dataloader)

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                total_loss += loss.item()

                # Calculate token-level prediction accuracy
                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = targets.view(-1)

                valid_mask = flat_targets != loss_fn.ignore_index
                if valid_mask.any():
                    active_logits = flat_logits[valid_mask]
                    active_targets = flat_targets[valid_mask]

                    preds = torch.argmax(active_logits, dim=-1)
                    correct_tokens += (preds == active_targets).sum().item()
                    total_tokens += active_targets.size(0)

        avg_loss = total_loss / max(1, total_batches)
        perplexity = PerplexityEvaluator.compute(avg_loss)
        accuracy = correct_tokens / max(1, total_tokens)

        # Restore model training status
        model.train()

        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
            "val_accuracy": accuracy,
        }
