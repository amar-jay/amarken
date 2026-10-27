"""Deterministic random numeric data for modules.example."""

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from amarken.constants import EXAMPLE_INPUT_SIZE
from amarken.datasets.config import DatasetConfig
from amarken.distributed import train_sampler


def build_dataset(config: DatasetConfig, *, seed: int = 42) -> tuple[Dataset, Dataset]:
    train_size, val_size = config.train_size, config.val_size
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(
        train_size + val_size,
        EXAMPLE_INPUT_SIZE,
        generator=generator,
    )
    targets = (inputs[:, 0] + inputs[:, 1] > 0).long()
    return (
        TensorDataset(inputs[:train_size], targets[:train_size]),
        TensorDataset(inputs[train_size:], targets[train_size:]),
    )


def build_dataloaders(
    config: DatasetConfig,
    train: Dataset,
    validation: Dataset,
    *,
    batch_size: int,
    tokenize: Callable | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Same signature as huggingface; numeric tensors need no tokenizer."""
    sampler = train_sampler(train)
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=0,
        ),
        DataLoader(validation, batch_size=batch_size, num_workers=0),
    )


def build_test_dataset(config: DatasetConfig, *, seed: int = 42) -> Dataset:
    """Use a separate random stream so test rows are independent of split sizes."""
    test_size = config.test_size
    generator = torch.Generator().manual_seed(seed + 1)
    inputs = torch.randn(test_size, EXAMPLE_INPUT_SIZE, generator=generator)
    targets = (inputs[:, 0] + inputs[:, 1] > 0).long()
    return TensorDataset(inputs, targets)
