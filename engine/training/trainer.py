# ruff: noqa: E501, C901
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from engine.tokenizer.service import TokenizerService
from engine.training.config.schema import ModelTrainingConfig
from engine.training.dataset.dataloader import ChatiiDataLoader
from engine.training.dataset.dataset import ChatiiDataset
from engine.training.events.bus import EventBus
from engine.training.events.hooks import TrainingEvents
from engine.training.exceptions import (
    CheckpointError,
    EarlyStoppingTrigger,
    NaNLossError,
)
from engine.training.gradient.accumulation import GradientAccumulator
from engine.training.gradient.clipping import GradientClipper
from engine.training.gradient.scaler import LossScaler
from engine.training.loss.cross_entropy import CrossEntropyLoss
from engine.training.manifest import TrainingManifest
from engine.training.optimizer.factory import OptimizerFactory
from engine.training.scheduler.factory import SchedulerFactory
from engine.training.training_state import TrainingState
from engine.training.validation.evaluator import SFTValEvaluator


class BaseTrainer:
    """Abstract base trainer class standardizing model training pipelines."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerService,
        config: ModelTrainingConfig,
        train_dataset: ChatiiDataset,
        val_dataset: Optional[ChatiiDataset] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = torch.device(
            "cuda"
            if config.device == "cuda"
            or (config.device == "auto" and torch.cuda.is_available())
            else "mps"
            if config.device == "mps"
            or (config.device == "auto" and torch.backends.mps.is_available())
            else "cpu"
        )
        self.model.to(self.device)

    def train(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> Dict[str, Any]:
        raise NotImplementedError

    def save_checkpoint(self, path: str) -> None:
        raise NotImplementedError

    def resume_from_checkpoint(self, path: str) -> None:
        raise NotImplementedError


class SFTTrainer(BaseTrainer):
    """
    Standard Supervised Fine-Tuning (SFT) trainer implementing
    lifecycle hooks, mixed precision, and checkpoint signature validation.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerService,
        config: ModelTrainingConfig,
        train_dataset: ChatiiDataset,
        val_dataset: Optional[ChatiiDataset] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> None:
        super().__init__(model, tokenizer, config, train_dataset, val_dataset)

        self.state = TrainingState()
        self.event_bus = EventBus()

        # Build dataloaders
        self.train_loader = ChatiiDataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        self.val_loader = (
            ChatiiDataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
            if val_dataset is not None
            else None
        )

        # Build loss, optimizer, and scheduler
        self.loss_fn = CrossEntropyLoss(ignore_index=-100)
        self.optimizer = OptimizerFactory.create(
            config.optimizer.name,
            self.model,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            eps=config.optimizer.eps,
        )

        # Estimate total training steps
        total_steps = len(self.train_loader) * config.epochs
        self.scheduler = SchedulerFactory.create(
            config.scheduler.name,
            self.optimizer,
            base_lr=config.optimizer.lr,
            total_steps=total_steps,
            warmup_steps=config.scheduler.warmup_steps,
            min_lr=config.scheduler.min_lr,
        )

        # Gradient management components
        self.grad_accumulator = GradientAccumulator(config.gradient_accumulation_steps)
        self.loss_scaler = LossScaler()

        # Mixed precision settings
        self.use_amp = config.mixed_precision in ["fp16", "bf16"]
        self.amp_dtype = (
            torch.float16
            if config.mixed_precision == "fp16"
            else torch.bfloat16
            if config.mixed_precision == "bf16"
            else torch.float32
        )

        # Register callbacks via CallbackHandler if provided
        if callbacks:
            from engine.training.callbacks.base import CallbackHandler

            self.callback_handler = CallbackHandler(callbacks, self.event_bus)
        else:
            self.callback_handler = None

    def train(self) -> None:
        """Runs the main supervised fine-tuning training loop."""
        self.event_bus.publish(TrainingEvents.TRAIN_START, self.state, trainer=self)

        try:
            for epoch in range(self.state.epoch, self.config.epochs):
                self.state.epoch = epoch
                self.event_bus.publish(
                    TrainingEvents.EPOCH_START, self.state, trainer=self
                )

                for _batch_idx, (inputs, targets) in enumerate(self.train_loader):
                    self.event_bus.publish(
                        TrainingEvents.STEP_START, self.state, trainer=self
                    )

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # 1. Forward pass under autocast context
                    with torch.amp.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=self.use_amp,
                    ):
                        logits = self.model(inputs)
                        loss = self.loss_fn(logits, targets)
                        # Normalize loss value by accumulation steps
                        loss = loss / self.grad_accumulator.accumulation_steps

                    if torch.isnan(loss) or torch.isinf(loss):
                        raise NaNLossError(
                            "Gradients calculation halted: Loss is NaN/Inf."
                        )

                    # 2. Backward pass with loss scaling
                    self.loss_scaler.scale_loss(loss).backward()
                    self.grad_accumulator.increment()

                    # 3. Optimizer update step after gradient accumulation
                    if self.grad_accumulator.should_step():
                        if self.config.grad_clip > 0.0:
                            if self.loss_scaler.native_scaler is not None:
                                self.loss_scaler.native_scaler.unscale_(self.optimizer)
                            GradientClipper.clip_grad_norm(
                                [p.data for p in self.model.parameters()],
                                self.config.grad_clip,
                            )

                        step_ok = self.loss_scaler.step(self.optimizer)
                        if step_ok:
                            lr = self.scheduler.step(self.state.global_step)
                            self.state.learning_rate = lr

                        self.optimizer.zero_grad()
                        self.state.global_step += 1

                    # Update step metrics state
                    tokens = inputs.size(0) * inputs.size(1)
                    self.state.update(
                        loss=loss.item() * self.grad_accumulator.accumulation_steps,
                        lr=self.state.learning_rate,
                        tokens=tokens,
                        examples=inputs.size(0),
                    )

                    self.event_bus.publish(
                        TrainingEvents.STEP_END, self.state, trainer=self
                    )

                    if (
                        self.config.max_steps is not None
                        and self.state.global_step >= self.config.max_steps
                    ):
                        break

                self.event_bus.publish(
                    TrainingEvents.EPOCH_END, self.state, trainer=self
                )

                if (
                    self.config.max_steps is not None
                    and self.state.global_step >= self.config.max_steps
                ):
                    break

        except EarlyStoppingTrigger as e:
            print(f"\n{e}")
        finally:
            self.event_bus.publish(TrainingEvents.TRAIN_END, self.state, trainer=self)

    def evaluate(self) -> Dict[str, Any]:
        """Runs validation evaluations and publishes VAL_END event."""
        if self.val_loader is None:
            return {}

        self.event_bus.publish(TrainingEvents.VAL_START, self.state, trainer=self)
        metrics = SFTValEvaluator.evaluate(
            self.model, self.val_loader, self.loss_fn, self.device
        )
        self.state.metrics.update(metrics)
        self.event_bus.publish(TrainingEvents.VAL_END, self.state, trainer=self)
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Serializes model parameters and training state, writing SHA256 checksums."""
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))

        # Save training state parameters
        with open(os.path.join(path, "trainer.json"), "w", encoding="utf-8") as f:
            json.dump(self.state.state_dict(), f, indent=2)

        # Save config dump
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            f.write(self.config.model_dump_json(indent=2))

        # Generate dataset fingerprints
        fingerprint = TrainingManifest.generate_fingerprint(
            self.train_dataset.file_path
        )
        manifest_path = os.path.join(path, "manifest.json")
        TrainingManifest.write_manifest(
            manifest_path,
            model_name="chatii-transformer",
            dataset_fingerprint=fingerprint,
            tokens_seen=self.state.tokens_seen,
            epochs=self.state.epoch + 1,
            checkpoint=os.path.basename(path),
            status="running",
        )

        # Generate SHA256 checksum validation directory
        checksums = {}
        for name in os.listdir(path):
            if name == "checksum.sha256":
                continue
            file_path = os.path.join(path, name)
            h = hashlib.sha256()
            with open(file_path, "rb") as f_bin:
                while chunk := f_bin.read(8192):
                    h.update(chunk)
            checksums[name] = h.hexdigest()

        with open(
            os.path.join(path, "checksum.sha256"), "w", encoding="utf-8"
        ) as f_sum:
            json.dump(checksums, f_sum, indent=2)

        # Rotate checkpoints to keep only top N step_ directories
        from engine.training.checkpoint.manager import CheckpointManager

        chk_mgr = CheckpointManager(
            os.path.dirname(path), keep_top_n=self.config.checkpoint.keep_top_n
        )
        chk_mgr.rotate_checkpoints()

        self.event_bus.publish(TrainingEvents.CHECKPOINT_SAVE, self.state, trainer=self)

    def resume_from_checkpoint(self, path: str) -> None:
        """Loads serialized training parameters, verifying checksum signatures."""
        chk_path = Path(path)
        if not chk_path.exists():
            raise CheckpointError(f"Checkpoint folder '{path}' does not exist.")

        # 1. Verify checksum signatures
        checksum_path = chk_path / "checksum.sha256"
        if not checksum_path.exists():
            raise CheckpointError("Missing required checksum.sha256 validation file.")

        with open(checksum_path, "r", encoding="utf-8") as f:
            checksums = json.load(f)

        for name, expected_hash in checksums.items():
            file_path = chk_path / name
            if not file_path.exists():
                raise CheckpointError(f"Missing expected checkpoint file: '{name}'")

            h = hashlib.sha256()
            with open(file_path, "rb") as f_bin:
                while chunk := f_bin.read(8192):
                    h.update(chunk)

            if h.hexdigest() != expected_hash:
                raise CheckpointError(
                    f"Checksum mismatch for checkpoint file: '{name}'."
                )

        # 2. Deserialization
        self.model.load_state_dict(
            torch.load(
                chk_path / "model.pt", map_location=self.device, weights_only=True
            )
        )
        self.optimizer.load_state_dict(
            torch.load(
                chk_path / "optimizer.pt", map_location=self.device, weights_only=True
            )
        )

        with open(chk_path / "trainer.json", "r", encoding="utf-8") as f:
            state_data = json.load(f)
        self.state.load_state_dict(state_data)
