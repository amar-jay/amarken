"""Single-device, DDP and FSDP2 training with epoch-boundary checkpoints."""

import os
import random
import sys
import warnings
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from amarken import distributed
from amarken.constants import BASES
from amarken.logger import CheckpointConfig, WandbConfig, WandbLogger
from amarken.utils import prepare_run, prepare_training_precision, seed_everything


@dataclass
class TrainingConfig:
    model_id: str = BASES[0]
    seed: int = 42
    learning_rate: float = 3e-4
    batch_size: int = 16
    train_size: int = 128
    val_size: int = 32
    test_size: int = 32
    epochs: int = 3
    device: str = "cpu"
    strategy: str = "auto"
    precision: str = "float32"
    run_name: str = "training"
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoints: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(backend="local")
    )

    def __post_init__(self):
        if isinstance(self.wandb, dict):
            self.wandb = WandbConfig(**self.wandb)
        if isinstance(self.checkpoints, dict):
            self.checkpoints = CheckpointConfig(**self.checkpoints)
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")
        for name in ("train_size", "val_size", "test_size"):
            size = getattr(self, name)
            if type(size) is not int or size < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.strategy not in {"auto", "ddp", "fsdp2"}:
            raise ValueError("strategy must be auto, ddp or fsdp2")
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("precision must be float32, float16 or bfloat16")
        if self.precision == "float16" and torch.device(self.device).type != "cuda":
            raise ValueError("float16 training requires CUDA")
        if self.checkpoints.backend == "wandb" and not self.wandb.enabled:
            raise ValueError("W&B checkpoint storage requires wandb.enabled=true")
        if (
            not self.run_name
            or Path(self.run_name).name != self.run_name
            or self.run_name in {".", ".."}
        ):
            raise ValueError("run_name must be a single directory name")
        if self.strategy == "fsdp2" and self.device == "cpu":
            os.environ.setdefault("OMP_NUM_THREADS", "1")


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
        config: TrainingConfig,
        *,
        validation_loader=None,
        optimizer=None,
        scheduler=None,
        logger=None,
        run_label: str | None = None,
        fsdp_modules: Callable | None = None,
    ):
        if distributed.launched() and not distributed.active():
            raise ValueError(
                "Initialize distributed.training_process before constructing the trainer"
            )
        self.config = config
        self.device = torch.device(config.device)
        self.strategy = distributed.resolve_strategy(config.strategy, config.precision)
        prepare_training_precision(model, config.precision)
        if self.strategy == "fsdp2":
            if not distributed.active():
                raise ValueError(
                    "Initialize training_process before constructing an FSDP2 trainer"
                )
            if optimizer is not None or scheduler is not None:
                raise ValueError(
                    "FSDP2 constructs its optimizer after sharding; external optimizers/schedulers are unsupported"
                )
            if fsdp_modules is None:
                raise ValueError(
                    "FSDP2 requires the selected model's fsdp_modules hook"
                )
            self.model = distributed.shard_model(
                model, fsdp_modules, self.device, config.precision
            )
        else:
            self.model = model.to(self.device)
            if self.strategy == "ddp":
                device_ids = [self.device.index] if self.device.type == "cuda" else None
                self.model = DistributedDataParallel(self.model, device_ids=device_ids)
        if distributed.active():
            seed_everything(config.seed + distributed.rank())
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
        if run_label is not None and (
            not run_label
            or Path(run_label).name != run_label
            or run_label in {".", ".."}
        ):
            raise ValueError("Run name must be a single directory name")
        self.run_label = run_label
        self._resumed = False
        self._directory_prepared = False
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

    def state_dict(self, *, training_state=None):
        if self.strategy == "fsdp2" and training_state is None:
            raise ValueError(
                "FSDP2 state capture is collective; pass distributed.full_state(model, optimizer)"
            )
        if training_state is None:
            training_state = (
                distributed.unwrap(self.model).state_dict(),
                self.optimizer.state_dict(),
            )
        model_state, optimizer_state = training_state
        return {
            "format_version": 1,
            "run_id": self.run_id,
            "run_label": self.run_label,
            "best_metric": self.best_metric,
            "model": model_state,
            "strategy": self.strategy,
            "world_size": distributed.world_size(),
            "rank_rng": getattr(self, "_rank_rng", None),
            "optimizer": optimizer_state,
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
                torch.cuda.get_rng_state_all()
                if self.device.type == "cuda" and not distributed.active()
                else None
            ),
        }

    def resume(self, path: str | Path):
        restored = {}
        if self.strategy == "fsdp2":

            def read_checkpoint():
                restored.update(torch.load(path, map_location="cpu", weights_only=True))
                return {
                    key: value
                    for key, value in restored.items()
                    if key not in {"model", "optimizer"}
                }

            state = distributed.primary_call(read_checkpoint)
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("world_size", 1) != distributed.world_size():
            raise ValueError("Resume requires the same world size")
        previous_strategy = state.get(
            "strategy", "ddp" if state.get("world_size", 1) > 1 else "single"
        )
        if previous_strategy != self.strategy:
            raise ValueError("Resume requires the same strategy")
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
        run_label = state.get("run_label")
        if run_label is not None and (
            not run_label
            or Path(run_label).name != run_label
            or run_label in {".", ".."}
        ):
            raise ValueError("Invalid checkpoint run name")
        self.run_label = run_label
        self.run_id = state.get("run_id", self.run_id)
        if Path(self.run_id).name != self.run_id or self.run_id in {".", "..", ""}:
            raise ValueError("Invalid checkpoint run ID")
        self._resumed = True
        if self.strategy == "fsdp2":
            distributed.restore_full_state(
                self.model,
                self.optimizer,
                restored.get("model", {}),
                restored.get("optimizer", {}),
            )
        else:
            distributed.unwrap(self.model).load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.epoch, self.global_step = state["epoch"], state["global_step"]
        rng = state
        if state.get("rank_rng") is not None:
            rng = state["rank_rng"][distributed.rank()]
        random.setstate(rng["python_rng"])
        torch.set_rng_state(rng["torch_rng"])
        if rng.get("cuda_local_rng") is not None:
            torch.cuda.set_rng_state(rng["cuda_local_rng"], self.device)
        elif rng["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda_rng"])

    def _gather_rng(self):
        cuda_rng = None
        if self.device.type == "cuda":
            cuda_rng = torch.cuda.get_rng_state(self.device)
        rng = {
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": None,
            "cuda_local_rng": cuda_rng,
        }
        if distributed.active():
            states = [None] * distributed.world_size()
            dist.all_gather_object(states, rng)
            self._rank_rng = states

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
                count = len(batch[1])
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
        def prepare():
            prepare_run(self)
            return self.run_id, self.run_label

        self.run_id, self.run_label = distributed.primary_call(prepare)
        stack = ExitStack()
        original_logger = self.logger

        def open_logger():
            if self.logger is None:
                self.logger = stack.enter_context(
                    WandbLogger(
                        enabled=self.config.wandb.enabled,
                        mode=self.config.wandb.mode,
                        name=self.run_label or self.run_id,
                        config=asdict(self.config),
                        checkpoints=self.config.checkpoints,
                    )
                )

        try:
            distributed.primary_call(open_logger)
            history = self._fit()
            distributed.primary_call(lambda: self.logger.flush_checkpoints())
            distributed.primary_call(stack.close)
            return history
        except BaseException:
            if distributed.primary():
                try:
                    if self.logger is not None:
                        self.logger.flush_checkpoints()
                    stack.__exit__(*sys.exc_info())
                except Exception as error:  # noqa: BLE001 - preserve original failure
                    warnings.warn(
                        f"Checkpoint cleanup failed: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            raise
        finally:
            self.logger = original_logger

    @property
    def checkpoint_prefix(self):
        """Use stable paths or one named/UUID directory per run."""
        if self.config.checkpoints.overwrite:
            return self.config.run_name
        return f"{self.config.run_name}/{self.run_label or self.run_id}"

    def _save_checkpoints(self, metrics):
        settings = self.config.checkpoints
        filenames = []
        if settings.monitor not in metrics:
            raise ValueError(f"Checkpoint monitor {settings.monitor!r} is unavailable")
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
        if not filenames:
            return
        training_state = (
            distributed.full_state(self.model, self.optimizer)
            if self.strategy == "fsdp2"
            else None
        )

        def submit():
            snapshot = self.state_dict(training_state=training_state)
            for filename, aliases in filenames:
                artifact = self.logger.save_checkpoint(
                    snapshot,
                    f"{self.checkpoint_prefix}/{filename}",
                    name=f"model-{self.run_id}",
                    aliases=aliases,
                    # Update best/latest within the selected run directory.
                    overwrite=True,
                )
                if (
                    not settings.async_save
                    and settings.backend == "wandb"
                    and artifact is not None
                ):
                    artifact.wait()

        distributed.primary_call(submit)

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
            self._gather_rng()

            distributed.primary_call(
                lambda metrics=metrics: self.logger.log(
                    metrics, global_step=self.global_step
                )
            )
            self._save_checkpoints(metrics)
            history.append(metrics)
        return history
