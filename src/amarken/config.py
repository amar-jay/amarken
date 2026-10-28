"""Shared configuration for training, inference, and evaluation."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    mode: str = "offline"

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("wandb.enabled must be a boolean")
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")


@dataclass(frozen=True)
class CheckpointConfig:
    """Local checkpoints with optional W&B or S3-compatible replication.

    S3 credentials use boto3's normal environment/profile/IAM credential chain.
    Use a distinct prefix (or filenames) for independent training runs.
    """

    async_save: bool = False
    max_pending: int = 2
    policy: str = "default"
    overwrite: bool = True
    monitor: str = "val/loss"
    mode: str = "min"
    backend: str = "local"
    bucket: str | None = None
    prefix: str = "checkpoints"
    endpoint_url: str | None = None
    region_name: str | None = None

    def __post_init__(self):
        if type(self.async_save) is not bool:
            raise ValueError("async_save must be a boolean")
        if type(self.max_pending) is not int or self.max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        if type(self.overwrite) is not bool:
            raise ValueError("overwrite must be a boolean")
        if self.policy not in {"best", "default"}:
            raise ValueError("Checkpoint policy must be best or default")
        if self.mode not in {"min", "max"}:
            raise ValueError("Checkpoint mode must be min or max")
        if not self.monitor:
            raise ValueError("Checkpoint monitor must be nonempty")
        if self.backend not in {"local", "wandb", "s3"}:
            raise ValueError("Checkpoint backend must be local, wandb, or s3")
        if self.backend == "s3" and not self.bucket:
            raise ValueError("An S3 bucket is required for checkpoint storage")


@dataclass
class DatasetConfig:
    id: str
    train_size: int = 128
    val_size: int = 32
    test_size: int = 32
    config: str | None = None
    test_fraction: float = 0.5
    train_split: str = "train"
    validation_split: str = "validation"
    text_column: str = "sentence"
    label_column: str = "label"
    input_column: str | None = None

    def __post_init__(self):
        if self.input_column is not None and (
            not isinstance(self.input_column, str) or not self.input_column.strip()
        ):
            raise ValueError("dataset.input_column must be a nonempty string or null")
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("dataset.id must be a nonempty string")
        if self.config is not None and (
            not isinstance(self.config, str) or not self.config.strip()
        ):
            raise ValueError("dataset.config must be a nonempty string or null")
        for name in ("train_split", "validation_split", "text_column", "label_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"dataset.{name} must be a nonempty string")
        for name in ("train_size", "val_size", "test_size"):
            size = getattr(self, name)
            if type(size) is not int or size < 1:
                raise ValueError(f"dataset.{name} must be a positive integer")
        if (
            type(self.test_fraction) not in {int, float}
            or not 0 < self.test_fraction < 1
        ):
            raise ValueError(
                "dataset.test_fraction must be between 0 and 1 (exclusive)"
            )


@dataclass
class FineTuningConfig:
    """Optional causal LM settings; omit the section for full fine-tuning."""

    mode: str = "full"
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    def __post_init__(self):
        if self.mode not in {"full", "lora"}:
            raise ValueError("fine_tuning.mode must be full or lora")
        if type(self.r) is not int or self.r < 1 or self.alpha <= 0:
            raise ValueError("LoRA r and alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if (
            not isinstance(self.target_modules, (list, tuple))
            or not self.target_modules
        ):
            raise ValueError("LoRA target_modules must be a nonempty list")
        if any(
            not isinstance(name, str) or not name.strip()
            for name in self.target_modules
        ):
            raise ValueError("LoRA target_modules must contain nonempty names")
        self.target_modules = tuple(self.target_modules)


@dataclass
class RLConfig:
    """Single-device GRPO settings. Batch size counts prompts, not completions."""

    group_size: int = 4
    max_new_tokens: int = 16
    eval_max_new_tokens: int | None = None
    temperature: float = 1.0
    clip_epsilon: float = 0.2
    beta: float = 0.02
    update_epochs: int = 2
    max_grad_norm: float = 1.0
    initial_checkpoint: str | None = None

    def __post_init__(self):
        for key, minimum in (
            ("group_size", 2),
            ("max_new_tokens", 1),
            ("update_epochs", 1),
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < minimum:
                raise ValueError(f"rl.{key} must be an integer >= {minimum}")
        if self.eval_max_new_tokens is not None and (
            type(self.eval_max_new_tokens) is not int or self.eval_max_new_tokens < 1
        ):
            raise ValueError(
                "rl.eval_max_new_tokens must be a positive integer or null"
            )
        for key in ("temperature", "clip_epsilon", "beta", "max_grad_norm"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"rl.{key} must be finite")
        if self.temperature <= 0 or self.max_grad_norm <= 0 or self.beta < 0:
            raise ValueError(
                "RL temperature/gradient norm must be positive and beta nonnegative"
            )
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("rl.clip_epsilon must be between 0 and 1")
        if self.initial_checkpoint is not None and (
            not isinstance(self.initial_checkpoint, str)
            or not self.initial_checkpoint.strip()
        ):
            raise ValueError("rl.initial_checkpoint must be a nonempty path or null")

    @property
    def evaluation_generation(self):
        """Settings passed to the task's generation-based validation hook."""
        return {
            "max_new_tokens": self.max_new_tokens
            if self.eval_max_new_tokens is None
            else self.eval_max_new_tokens
        }


@dataclass
class Config:
    model_id: str
    dataset: DatasetConfig
    seed: int = 42
    learning_rate: float = 3e-4
    batch_size: int = 16
    epochs: int = 3
    device: str = "cpu"
    strategy: str = "auto"
    precision: str = "float32"
    run_name: str = "training"
    fine_tuning: FineTuningConfig | None = None
    rl: RLConfig | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoints: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(backend="local")
    )

    def __post_init__(self):
        if isinstance(self.rl, dict):
            self.rl = RLConfig(**self.rl)
        if self.rl is not None and not isinstance(self.rl, RLConfig):
            raise TypeError("rl must be an RLConfig or dictionary")
        if isinstance(self.fine_tuning, dict):
            self.fine_tuning = FineTuningConfig(**self.fine_tuning)
        if self.fine_tuning is not None and not isinstance(
            self.fine_tuning, FineTuningConfig
        ):
            raise TypeError("fine_tuning must be a FineTuningConfig or dictionary")
        if isinstance(self.wandb, dict):
            self.wandb = WandbConfig(**self.wandb)
        if self.rl is not None and (
            self.fine_tuning is None or self.fine_tuning.mode != "lora"
        ):
            raise ValueError("The initial GRPO trainer requires fine_tuning.mode=lora")
        if isinstance(self.checkpoints, dict):
            self.checkpoints = CheckpointConfig(**self.checkpoints)
        if isinstance(self.dataset, dict):
            self.dataset = DatasetConfig(**self.dataset)
        if not isinstance(self.dataset, DatasetConfig):
            raise TypeError("dataset must be a DatasetConfig or dictionary")
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")
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
