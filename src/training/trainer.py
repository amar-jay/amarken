"""One exact-resumable training loop shared by every Amarken causal LM."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Literal

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from src.models.common import AmarkenCausalLM
from .data import PackedSequenceDataset
from .diagnostics import TernaryTransitionTracker, gradient_health, ternary_statistics
from .bit_optimizer import OptimizerConfig, create_optimizer


@dataclass(frozen=True)
class TrainerConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    seed: int = 2026
    log_every_steps: int = 10
    checkpoint_every_steps: int = 1_000
    output_dir: Path = Path("runs/default")
    lr_schedule: Literal["constant", "cosine"] = "constant"
    warmup_steps: int = 0
    total_steps: int | None = None
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        if (
            min(
                self.batch_size,
                self.gradient_accumulation_steps,
                self.log_every_steps,
                self.checkpoint_every_steps,
            )
            < 1
        ):
            raise ValueError(
                "batch, accumulation, log, and checkpoint intervals must be positive"
            )
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.warmup_steps < 0 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError(
                "warmup_steps must be nonnegative and min_lr_ratio in [0,1]"
            )
        if self.lr_schedule == "cosine" and (
            self.total_steps is None or self.total_steps <= self.warmup_steps
        ):
            raise ValueError("cosine schedule requires total_steps > warmup_steps")


@dataclass
class TrainerState:
    update_step: int = 0
    consumed_micro_batches: int = 0
    consumed_tokens: int = 0
    epoch: int = 0
    block_offset: int = 0
    clipped_updates: int = 0


class Trainer:
    """Shared AdamW/AMP trainer whose checkpoints resume at optimizer boundaries."""

    FORMAT_VERSION = 2
    # These settings alter batch selection or floating-point updates and must be
    # identical after resume. Output/log/checkpoint intervals may safely change.
    TRAJECTORY_CONFIG_FIELDS = (
        "batch_size",
        "gradient_accumulation_steps",
        "precision",
        "gradient_checkpointing",
        "max_grad_norm",
        "seed",
        "lr_schedule",
        "warmup_steps",
        "total_steps",
        "min_lr_ratio",
    )

    def __init__(
        self,
        model: AmarkenCausalLM,
        dataset: PackedSequenceDataset,
        config: TrainerConfig,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if config.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 training requires CUDA; use bf16 or fp32 on CPU")
        if dataset.sequence_length > model.config.max_position_embeddings:
            raise ValueError("packed sequence length exceeds model context")
        self.model = model.to(self.device)
        self.model.set_gradient_checkpointing(config.gradient_checkpointing)
        self.dataset = dataset
        self.config = config
        self.optimizer_config = optimizer_config
        self.optimizer = create_optimizer(model, optimizer_config)
        # Scaling is needed for FP16's narrow exponent, not BF16. Passing the
        # device string follows the unified torch.amp API and avoids deprecated
        # torch.cuda.amp constructors.
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
        self.state = TrainerState()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.config.output_dir / "metrics.jsonl"
        self._dataset_fingerprint = self._fingerprint_dataset()
        self._ternary_transitions = TernaryTransitionTracker(self.model)

    def _fingerprint_dataset(self) -> str:
        digest = hashlib.sha256()
        for block in self.dataset.blocks:
            # Decimal delimiters avoid dtype/endianness dependence in provenance.
            digest.update((",".join(map(str, block.input_ids)) + "|").encode())
            digest.update((",".join(map(str, block.labels)) + "\n").encode())
        return digest.hexdigest()

    def _epoch_order(self, epoch: int) -> list[int]:
        # A fresh CPU generator makes order a pure function of seed+epoch and does
        # not consume the model/dropout RNG that exact resume must restore.
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed + epoch)
        return torch.randperm(len(self.dataset), generator=generator).tolist()

    def _next_batch(self) -> dict[str, Tensor]:
        order = self._epoch_order(self.state.epoch)
        if self.state.block_offset >= len(order):
            self.state.epoch += 1
            self.state.block_offset = 0
            order = self._epoch_order(self.state.epoch)
        end = min(self.state.block_offset + self.config.batch_size, len(order))
        indices = order[self.state.block_offset : end]
        self.state.block_offset = end
        self.state.consumed_micro_batches += 1
        return self.dataset.batch(indices, self.device)

    def _autocast(self):
        if self.config.precision == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def _forward_loss(self, batch: dict[str, Tensor]) -> Tensor:
        attention_mask: Tensor | None = batch["attention_mask"]
        segment_ids: Tensor | None = batch["segment_ids"]
        # A genuinely single-segment, unpadded batch can use native causal SDPA;
        # multi-document packing must retain the dense segment isolation mask.
        one_segment = all(row[row >= 0].unique().numel() <= 1 for row in segment_ids)
        if bool(attention_mask.all()) and one_segment:
            attention_mask = None
            segment_ids = None
        loss = self.model(
            batch["input_ids"],
            attention_mask=attention_mask,
            labels=batch["labels"],
            segment_ids=segment_ids,
        ).loss
        if loss is None:
            raise RuntimeError("model did not return training loss")
        return loss

    def _set_learning_rates(self) -> None:
        """Set the next update LR as a pure function of completed update count."""
        next_step = self.state.update_step + 1
        if self.config.warmup_steps and next_step <= self.config.warmup_steps:
            multiplier = next_step / self.config.warmup_steps
        elif self.config.lr_schedule == "cosine":
            decay_steps = self.config.total_steps - self.config.warmup_steps
            progress = min(1.0, (next_step - self.config.warmup_steps) / decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            multiplier = (
                self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine
            )
        else:
            multiplier = 1.0
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier

    def _log(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def train(self, updates: int) -> list[dict]:
        """Run `updates` successful optimizer steps and return emitted log records."""
        if updates < 0:
            raise ValueError("updates cannot be negative")
        target = self.state.update_step + updates
        emitted = []
        self.model.train()
        while self.state.update_step < target:
            started = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            self._set_learning_rates()
            micro_batches = [
                self._next_batch()
                for _ in range(self.config.gradient_accumulation_steps)
            ]
            supervised = [
                int((batch["labels"][:, 1:] != -100).sum()) for batch in micro_batches
            ]
            supervised_total = sum(supervised)
            if supervised_total == 0:
                raise ValueError(
                    "gradient accumulation window has no assistant-token targets"
                )
            weighted_loss = 0.0
            for batch, target_tokens in zip(micro_batches, supervised):
                if target_tokens == 0:
                    continue
                with self._autocast():
                    loss = self._forward_loss(batch)
                    contribution = loss * (target_tokens / supervised_total)
                self.scaler.scale(contribution).backward()
                weighted_loss += float(loss.detach()) * target_tokens
            # Diagnostics and clipping must observe true gradients, so AMP
            # unscales once after all accumulation and before either operation.
            self.scaler.unscale_(self.optimizer)
            # Full tensor scans are valuable diagnostics but expensive for Bit;
            # collect them only on emitted log steps, never silently every update.
            diagnostics_due = (
                self.state.update_step + 1
            ) % self.config.log_every_steps == 0
            health = gradient_health(self.model) if diagnostics_due else {}
            bit_stats = ternary_statistics(self.model) if diagnostics_due else {}
            clipped_norm = float(
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            )
            if clipped_norm > self.config.max_grad_norm:
                self.state.clipped_updates += 1
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            skipped = self.scaler.get_scale() < old_scale
            tokens_this_step = sum(
                int(batch["attention_mask"].sum()) for batch in micro_batches
            )
            self.state.consumed_tokens += tokens_this_step
            if skipped:
                # Data cursor advances because those batches were inspected, but a
                # skipped overflow is not an optimizer update and cannot satisfy
                # the caller's requested update count.
                continue
            self.state.update_step += 1
            # When logging is sparse this deliberately measures churn since the
            # previous observation, which is both cheaper and more interpretable.
            transition_stats = (
                self._ternary_transitions.measure(self.model) if diagnostics_due else {}
            )
            record = {
                "step": self.state.update_step,
                "loss": weighted_loss / supervised_total,
                "learning_rates": {
                    group["group_name"]: group["lr"]
                    for group in self.optimizer.param_groups
                },
                "tokens": tokens_this_step,
                "supervised_tokens": supervised_total,
                "consumed_tokens": self.state.consumed_tokens,
                "epoch": self.state.epoch,
                "block_offset": self.state.block_offset,
                "grad_norm_before_clip": clipped_norm,
                "grad_clip_coefficient": min(
                    1.0, self.config.max_grad_norm / max(clipped_norm, 1e-12)
                ),
                "gradient_clipping_fraction": self.state.clipped_updates
                / self.state.update_step,
                "amp_scale": self.scaler.get_scale(),
                "seconds": time.perf_counter() - started,
                **health,
                **bit_stats,
                **transition_stats,
            }
            if self.state.update_step % self.config.log_every_steps == 0:
                self._log(record)
                emitted.append(record)
            if self.state.update_step % self.config.checkpoint_every_steps == 0:
                self.save_checkpoint(
                    self.config.output_dir / f"step-{self.state.update_step:08d}.pt"
                )
        return emitted

    def _payload(self, metadata: dict | None = None) -> dict:
        payload = {
            "format_version": self.FORMAT_VERSION,
            "model_type": self.model.config.model_type,
            "model_config": asdict(self.model.config),
            "trainer_config": {
                **asdict(self.config),
                "output_dir": str(self.config.output_dir),
            },
            "optimizer_config": asdict(self.optimizer_config),
            "dataset_fingerprint": self._dataset_fingerprint,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "trainer_state": asdict(self.state),
            "python_rng_state": random.getstate(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        return payload

    def _trajectory_config(self) -> dict:
        return {
            field: getattr(self.config, field)
            for field in self.TRAJECTORY_CONFIG_FIELDS
        }

    def save_checkpoint(self, path: str | Path, metadata: dict | None = None) -> None:
        """Atomically persist every state that can affect the next optimizer step."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(self._payload(metadata), temporary)
        os.replace(temporary, destination)

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("unsupported trainer checkpoint format")
        if payload["model_type"] != self.model.config.model_type or payload[
            "model_config"
        ] != asdict(self.model.config):
            raise ValueError("checkpoint model/config does not match trainer")
        if payload["optimizer_config"] != asdict(self.optimizer_config):
            raise ValueError(
                "checkpoint optimizer configuration does not match trainer"
            )
        saved_trainer = payload["trainer_config"]
        if any(
            saved_trainer[field] != value
            for field, value in self._trajectory_config().items()
        ):
            raise ValueError(
                "checkpoint trajectory configuration does not match trainer"
            )
        if payload["dataset_fingerprint"] != self._dataset_fingerprint:
            raise ValueError("checkpoint dataset does not match trainer")
        self.model.load_state_dict(payload["model_state"], strict=True)
        # Anchor telemetry to restored weights; construction-time random weights
        # otherwise make the first post-resume transition meaningless.
        self._ternary_transitions.reset(self.model)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.scaler.load_state_dict(payload["scaler_state"])
        self.state = TrainerState(**payload["trainer_state"])
        random.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["cpu_rng_state"].cpu())
        if torch.cuda.is_available() and payload["cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
