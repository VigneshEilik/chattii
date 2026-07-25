# ruff: noqa: E501
import time
from typing import Any, Dict

from engine.training.trainer import SFTTrainer


class TrainerBenchmark:
    """Measures sequence processing throughput and optimizer update step speeds."""

    @staticmethod
    def run_benchmark(trainer: SFTTrainer, iterations: int = 10) -> Dict[str, Any]:
        """
        Runs training iterations, calculates throughput, and returns step performance logs.
        """
        trainer.model.train()
        loader_iter = iter(trainer.train_loader)

        # Warmup runs to stabilize CUDA kernels
        for _ in range(3):
            try:
                inputs, targets = next(loader_iter)
            except StopIteration:
                loader_iter = iter(trainer.train_loader)
                inputs, targets = next(loader_iter)

            inputs = inputs.to(trainer.device)
            targets = targets.to(trainer.device)

            logits = trainer.model(inputs)
            loss = trainer.loss_fn(logits, targets)
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

        start = time.perf_counter()
        total_tokens = 0

        for _ in range(iterations):
            try:
                inputs, targets = next(loader_iter)
            except StopIteration:
                loader_iter = iter(trainer.train_loader)
                inputs, targets = next(loader_iter)

            inputs = inputs.to(trainer.device)
            targets = targets.to(trainer.device)

            # Execution Step
            logits = trainer.model(inputs)
            loss = trainer.loss_fn(logits, targets)
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

            total_tokens += inputs.size(0) * inputs.size(1)

        end = time.perf_counter()
        elapsed = end - start

        return {
            "elapsed_seconds": round(elapsed, 4),
            "tokens_per_sec": round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0,
            "latency_ms_per_step": round((elapsed / iterations) * 1000, 4),
        }
