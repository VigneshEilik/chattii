import os
import tempfile

import torch
import torch.nn as nn

from engine.tokenizer.service import TokenizerService
from engine.training.dataset.dataloader import ChatiiDataLoader
from engine.training.dataset.dataset import ChatiiDataset
from engine.training.gradient.accumulation import GradientAccumulator
from engine.training.gradient.clipping import GradientClipper
from engine.training.gradient.scaler import LossScaler
from engine.training.loss.cross_entropy import CrossEntropyLoss
from engine.training.optimizer.adamw import AdamW
from engine.training.optimizer.filter import WeightDecayFilter
from engine.training.optimizer.sgd import SGD
from engine.training.scheduler.factory import SchedulerFactory


class MockModel(nn.Module):
    """Simple linear mock model to trace backpropagation and parameter updates."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mock transformer logits format: [batch, seq_len, vocab_size]
        return self.linear(torch.randn(x.size(0), x.size(1), 10, device=x.device))


def test_custom_loss() -> None:
    loss_fn = CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
    logits = torch.randn(2, 5, 10)
    targets = torch.randint(0, 10, (2, 5))
    targets[0, 1] = -100  # set ignore token ID
    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_optimizers_and_filter() -> None:
    model = MockModel()
    groups = WeightDecayFilter.get_parameter_groups(model, weight_decay=0.1)

    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0

    opt_adam = AdamW(groups, lr=1e-3)
    opt_sgd = SGD(groups, lr=1e-3)

    loss = model(torch.randint(0, 10, (2, 4))).sum()
    loss.backward()

    opt_adam.step()
    opt_sgd.step()


def test_schedulers() -> None:
    model = MockModel()
    opt = AdamW(model.parameters(), lr=1e-3)

    # Cosine scheduler with warmup
    sched = SchedulerFactory.create(
        "cosine",
        optimizer=opt,
        base_lr=1e-3,
        total_steps=100,
        warmup_steps=10,
        min_lr=1e-4,
    )

    # Validate warmup phase learning rate scaling
    lr_w = sched.step(5)
    assert lr_w < 1e-3

    # Validate decay phase learning rate scaling
    lr_d = sched.step(50)
    assert lr_d < 1e-3


def test_gradient_engine() -> None:
    model = MockModel()
    loss = model(torch.randint(0, 10, (2, 4))).sum()
    loss.backward()

    # Gradient Norm Clipping
    norm = GradientClipper.clip_grad_norm(model.parameters(), max_norm=1.0)
    assert norm >= 0.0

    # Gradient Accumulation
    accum = GradientAccumulator(accumulation_steps=2)
    assert not accum.should_step()
    accum.increment()
    assert accum.should_step()

    # Dynamic Loss Scaler
    scaler = LossScaler()
    scaled_loss = scaler.scale_loss(loss)
    assert scaled_loss.item() != loss.item()


def test_dataset_loader() -> None:
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as tmp:
        tmp.write("This is a simple sentence.\nAnother story is here.")
        tmp_name = tmp.name

    try:
        # Initialize and train a character tokenizer vocabulary
        tok = TokenizerService("character")
        tok.train(tmp_name)

        dataset = ChatiiDataset(tmp_name, tok, sequence_length=8)
        assert len(dataset) > 0

        loader = ChatiiDataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        assert len(batch) == 2
        assert batch[0].shape == (2, 8)
        assert batch[1].shape == (2, 8)
    finally:
        os.remove(tmp_name)
