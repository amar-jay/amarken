"""Deterministic random numeric data for modules.example."""

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from amarken.constants import (
    EXAMPLE_INPUT_SIZE,
    RANDOM_TRAIN_SIZE,
    RANDOM_VALIDATION_SIZE,
)


def build_dataset(*, seed: int = 42) -> tuple[Dataset, Dataset]:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(
        RANDOM_TRAIN_SIZE + RANDOM_VALIDATION_SIZE,
        EXAMPLE_INPUT_SIZE,
        generator=generator,
    )
    targets = (inputs[:, 0] + inputs[:, 1] > 0).long()
    return (
        TensorDataset(inputs[:RANDOM_TRAIN_SIZE], targets[:RANDOM_TRAIN_SIZE]),
        TensorDataset(inputs[RANDOM_TRAIN_SIZE:], targets[RANDOM_TRAIN_SIZE:]),
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
