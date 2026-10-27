"""Single-device, DDP and FSDP2 training with epoch-boundary checkpoints."""

from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from amarken.config import Config
from amarken.utils import checkpoint, distributed, run
from amarken.utils.helpers import batch_size

__all__ = ["Trainer"]


class Trainer:
    """Accept a scalar batch loss callback: loss_fn(model, batch, device).

    Resume is at epoch boundaries. For reproducible resume use num_workers=0,
    global PyTorch RNG shuffling and deterministic data transforms. Custom
    worker RNG and gradient accumulation are left to task-specific implementations.
    """

    def __init__(
        self,
        model,
        train_loader,
        loss_fn: Callable,
        config: Config,
        *,
        validation_loader=None,
        optimizer=None,
        scheduler=None,
        logger=None,
        run_label: str | None = None,
        fsdp_modules: Callable | None = None,
    ):
        run.initialize(self, config, logger=logger, run_label=run_label)
        self.model, self.device, self.strategy = distributed.prepare_model(
            model,
            config,
            optimizer=optimizer,
            scheduler=scheduler,
            fsdp_modules=fsdp_modules,
        )
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.loss_fn = loss_fn
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        )
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.precision == "float16"
        )

    def _autocast(self):
        return torch.autocast(
            self.device.type,
            dtype=getattr(torch, self.config.precision),
            enabled=self.config.precision != "float32",
        )

    def state_dict(self, *, training_state=None):
        return checkpoint.capture_state(self, training_state=training_state)

    def resume(self, path: str | Path):
        checkpoint.restore_state(self, path)

    def _epoch_loss(self, loader, *, training):
        self.model.train(training)
        total, samples = 0.0, 0
        phase = "Train" if training else "Validation"
        with tqdm(
            loader,
            desc=f"Epoch {self.epoch + 1}/{self.config.epochs} {phase}",
            unit="batch",
            dynamic_ncols=True,
            disable=None if distributed.primary() else True,
        ) as progress:
            for batch in progress:
                if training:
                    distributed.primary_call(
                        lambda: (
                            self.logger.check_checkpoint_errors()
                            if self.logger
                            else None
                        )
                    )
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training), self._autocast():
                    loss = self.loss_fn(
                        self.model if training else distributed.unwrap(self.model),
                        batch,
                        self.device,
                    )
                valid = loss.ndim == 0 and torch.isfinite(loss).item()
                if distributed.active() and (training or self.strategy == "fsdp2"):
                    flag = torch.tensor(int(valid), device=self.device)
                    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                    valid = bool(flag.item())
                if not valid:
                    raise ValueError("loss_fn must return a finite scalar loss")
                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.global_step += 1
                count = batch_size(batch)
                total += loss.detach().item() * count
                samples += count
                progress.set_postfix(loss=total / samples, refresh=False)
        if distributed.active() and (training or self.strategy == "fsdp2"):
            aggregate = torch.tensor(
                [total, samples], dtype=torch.float64, device=self.device
            )
            dist.all_reduce(aggregate)
            total, samples = aggregate.tolist()
        if not samples:
            raise ValueError("Dataloaders must contain at least one batch")
        return total / samples

    def fit(self):
        return run.fit(self, self._fit)

    @property
    def checkpoint_prefix(self):
        return run.checkpoint_prefix(self)

    def _save_checkpoints(self, metrics):
        return checkpoint.save_checkpoints(
            self,
            metrics,
            capture_training_state=(
                lambda: distributed.full_state(self.model, self.optimizer)
            )
            if self.strategy == "fsdp2"
            else None,
        )

    def _fit(self):
        settings = self.config.checkpoints
        available = {"train/loss", "epoch"}
        if self.validation_loader is not None:
            available.add("val/loss")
        if settings.monitor not in available:
            raise ValueError(f"Checkpoint monitor {settings.monitor!r} is unavailable")
        history = []

        def define_metrics():
            self.logger.define_metric("train/*", step_metric="global_step")
            self.logger.define_metric("val/*", step_metric="global_step")

        distributed.primary_call(define_metrics)
        while self.epoch < self.config.epochs:
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)
            metrics = {"train/loss": self._epoch_loss(self.train_loader, training=True)}
            if self.validation_loader is not None:
                # Match rank-zero buffers before evaluation and future training.
                if distributed.active():
                    for buffer in distributed.unwrap(self.model).buffers():
                        dist.broadcast(buffer, src=0)
                if self.strategy == "fsdp2":
                    metrics["val/loss"] = self._epoch_loss(
                        self.validation_loader, training=False
                    )
                else:
                    metrics["val/loss"] = distributed.primary_call(
                        lambda: self._epoch_loss(self.validation_loader, training=False)
                    )
            if self.scheduler is not None:
                self.scheduler.step()
            self.epoch += 1
            metrics["epoch"] = self.epoch
            checkpoint.gather_rng(self)

            distributed.primary_call(
                lambda metrics=metrics: self.logger.log(
                    metrics, global_step=self.global_step
                )
            )
            self._save_checkpoints(metrics)
            history.append(metrics)
        return history
