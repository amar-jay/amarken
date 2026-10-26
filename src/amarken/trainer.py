"""Single-device training with epoch-boundary checkpoints."""

import os
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import torch

from amarken.constants import BASES
from amarken.logger import CheckpointConfig, WandbLogger


@dataclass
class TrainingConfig:
    model_id: str = BASES[0]
    seed: int = 42
    learning_rate: float = 3e-4
    batch_size: int = 16
    epochs: int = 3
    device: str = "cpu"
    precision: str = "float32"
    run_name: str = "training"
    wandb_enabled: bool = False
    wandb_mode: str = "offline"
    checkpoints: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(backend="local")
    )

    def __post_init__(self):
        if isinstance(self.checkpoints, dict):
            self.checkpoints = CheckpointConfig(**self.checkpoints)
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("precision must be float32, float16 or bfloat16")
        if self.precision == "float16" and torch.device(self.device).type != "cuda":
            raise ValueError("float16 training requires CUDA")
        if self.checkpoints.backend == "wandb" and not self.wandb_enabled:
            raise ValueError("W&B checkpoint storage requires wandb_enabled")
        if (
            not self.run_name
            or Path(self.run_name).name != self.run_name
            or self.run_name in {".", ".."}
        ):
            raise ValueError("run_name must be a single directory name")


def seed_everything(seed: int) -> None:
    """Call before constructing models and dataloaders."""
    random.seed(seed)
    torch.manual_seed(seed)


class Trainer:
    """Accept a scalar batch loss callback: loss_fn(model, batch, device).

    Resume is at epoch boundaries. For reproducible resume use num_workers=0,
    global PyTorch RNG shuffling and deterministic data transforms. Custom
    samplers, worker RNG, distributed training and gradient accumulation are
    intentionally left to task-specific implementations.
    """

    def __init__(
        self,
        model,
        train_loader,
        loss_fn: Callable,
        config: TrainingConfig,
        *,
        validation_loader=None,
        optimizer=None,
        scheduler=None,
        logger=None,
    ):
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError(
                "Trainer supports a single process; distributed training needs a custom loop"
            )
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.loss_fn = loss_fn
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        )
        self.scheduler = scheduler
        self.logger = logger
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.precision == "float16"
        )
        self.run_id = uuid4().hex
        self.best_metric = None
        self.epoch = 0
        self.global_step = 0

    def _autocast(self):
        return torch.autocast(
            self.device.type,
            dtype=getattr(torch, self.config.precision),
            enabled=self.config.precision != "float32",
        )

    def state_dict(self):
        return {
            "format_version": 1,
            "run_id": self.run_id,
            "best_metric": self.best_metric,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "config": asdict(self.config),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }

    def resume(self, path: str | Path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["format_version"] != 1:
            raise ValueError("Unsupported checkpoint format")
        if (state["scheduler"] is None) != (self.scheduler is None):
            raise ValueError("Resume requires the same scheduler configuration")
        if state["config"]["precision"] != self.config.precision:
            raise ValueError("Resume requires the same precision")
        previous = state["config"]["checkpoints"]
        current = self.config.checkpoints
        if (previous.get("monitor", "val/loss"), previous.get("mode", "min")) == (
            current.monitor,
            current.mode,
        ):
            self.best_metric = state.get("best_metric")
        else:
            self.best_metric = None
        self.run_id = state.get("run_id", self.run_id)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.epoch, self.global_step = state["epoch"], state["global_step"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def _epoch_loss(self, loader, *, training):
        self.model.train(training)
        total, batches = 0.0, 0
        for batch in loader:
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), self._autocast():
                loss = self.loss_fn(self.model, batch, self.device)
            if loss.ndim != 0 or not torch.isfinite(loss).item():
                raise ValueError("loss_fn must return a finite scalar loss")
            if training:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.global_step += 1
            total += loss.detach().item()
            batches += 1
        if not batches:
            raise ValueError("Dataloaders must contain at least one batch")
        return total / batches

    def fit(self):
        if self.logger is None:
            with WandbLogger(
                enabled=self.config.wandb_enabled,
                mode=self.config.wandb_mode,
                config=asdict(self.config),
                checkpoints=self.config.checkpoints,
            ) as logger:
                self.logger = logger
                try:
                    return self._fit()
                finally:
                    self.logger = None
        return self._fit()

    @property
    def checkpoint_prefix(self):
        """Fresh training runs have separate namespaces, including in buckets."""
        return f"{self.config.run_name}/{self.run_id}"

    def _save_checkpoints(self, metrics):
        settings = self.config.checkpoints
        filenames = []
        if settings.policy == "all":
            if (
                self.epoch % settings.every_n_epochs == 0
                or self.epoch == self.config.epochs
            ):
                filenames.append(
                    (
                        f"epoch-{self.epoch:04d}.pt",
                        (f"epoch-{self.epoch:04d}", "latest"),
                    )
                )
        else:
            if settings.monitor not in metrics:
                raise ValueError(
                    f"Checkpoint monitor {settings.monitor!r} is unavailable"
                )
            value = float(metrics[settings.monitor])
            improved = self.best_metric is None or (
                value < self.best_metric
                if settings.mode == "min"
                else value > self.best_metric
            )
            if improved:
                self.best_metric = value
                filenames.append(("best.pt", ("best",)))
            if settings.policy == "default":
                filenames.append(("latest.pt", ("latest",)))
        for filename, aliases in filenames:
            artifact = self.logger.save_checkpoint(
                self.state_dict(),
                f"{self.checkpoint_prefix}/{filename}",
                name=f"model-{self.run_id}",
                aliases=aliases,
                overwrite=settings.policy != "all",
            )
            if settings.backend == "wandb" and artifact is not None:
                artifact.wait()

    def _fit(self):
        settings = self.config.checkpoints
        available = {"train/loss", "epoch"}
        if self.validation_loader is not None:
            available.add("val/loss")
        if settings.policy != "all" and settings.monitor not in available:
            raise ValueError(f"Checkpoint monitor {settings.monitor!r} is unavailable")
        history = []
        self.logger.define_metric("train/*", step_metric="global_step")
        self.logger.define_metric("val/*", step_metric="global_step")
        while self.epoch < self.config.epochs:
            metrics = {"train/loss": self._epoch_loss(self.train_loader, training=True)}
            if self.validation_loader is not None:
                metrics["val/loss"] = self._epoch_loss(
                    self.validation_loader, training=False
                )
            if self.scheduler is not None:
                self.scheduler.step()
            self.epoch += 1
            metrics["epoch"] = self.epoch
            self.logger.log(metrics, global_step=self.global_step)
            self._save_checkpoints(metrics)
            history.append(metrics)
        return history
