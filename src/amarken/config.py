"""Shared configuration for training, inference, and evaluation."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from amarken.datasets.config import DatasetConfig
from amarken.utils.logger import CheckpointConfig, WandbConfig


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
    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoints: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(backend="local")
    )

    def __post_init__(self):
        if isinstance(self.wandb, dict):
            self.wandb = WandbConfig(**self.wandb)
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
