"""Deterministic random numeric data for modules.example."""

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from amarken.constants import EXAMPLE_INPUT_SIZE


def build_dataset(
    *, seed: int = 42, train_size: int = 128, val_size: int = 32
) -> tuple[Dataset, Dataset]:
    if any(type(size) is not int or size < 1 for size in (train_size, val_size)):
        raise ValueError("Dataset sizes must be positive integers")
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
    train: Dataset,
    validation: Dataset,
    *,
    batch_size: int,
    tokenize: Callable | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Same signature as huggingface; numeric tensors need no tokenizer."""
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(validation, batch_size=batch_size, num_workers=0),
    )
