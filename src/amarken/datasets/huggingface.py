"""Hugging Face text-classification data for modules.classification."""

from collections.abc import Callable, Mapping
from math import ceil

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from amarken.constants import (
    DATA_CACHE,
    DATASET_REVISIONS,
)
from amarken.config import DatasetConfig
from amarken.utils.distributed import train_sampler


def _build_dataset(
    config: DatasetConfig, *, seed: int = 42, _test: bool = False
) -> tuple[Dataset, Dataset]:
    train, validation = load_dataset(
        config.id,
        name=config.config,
        revision=DATASET_REVISIONS.get(config.id),
        split=[config.train_split, config.validation_split],
        cache_dir=DATA_CACHE,
    )

    # A fixed partition seed keeps validation/test membership independent of
    # the run's sampling seed. SST-2's public test labels are unavailable.
    if len(validation) < 2:
        raise ValueError("Need at least two labeled rows for validation and test")
    validation = validation.shuffle(seed=0)
    test_count = min(
        len(validation) - 1, max(1, ceil(len(validation) * config.test_fraction))
    )
    boundary = len(validation) - test_count
    validation = validation.select(
        range(boundary, len(validation)) if _test else range(boundary)
    )

    def sample(dataset, size):
        required = {config.text_column, config.label_column}
        if not required.issubset(dataset.column_names):
            raise ValueError(f"Dataset requires columns {sorted(required)}")
        if len(dataset) == 0:
            raise ValueError("Dataset splits must not be empty")
        dataset = dataset.shuffle(seed=seed).select(range(min(size, len(dataset))))
        return dataset

    return (
        sample(train, 1 if _test else config.train_size),
        sample(validation, config.test_size if _test else config.val_size),
    )


def build_dataset(config: DatasetConfig, *, seed: int = 42) -> tuple[Dataset, Dataset]:
    return _build_dataset(config, seed=seed)


def build_dataloaders(
    config: DatasetConfig,
    train: Dataset,
    validation: Dataset,
    *,
    batch_size: int,
    tokenize: Callable | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Yield (tokenized_dictionary, class_ids), matching classification.loss_fn.

    Tokenization runs in the training process. The model's tokenize() handles
    padding, context length and tensor placement. Keep num_workers=0 because
    that callback may use a tokenizer and a model on an accelerator.
    """
    if not callable(tokenize):
        raise TypeError("The Hugging Face text dataloader requires model.tokenize")

    def collate(samples):
        inputs = tokenize([sample[config.text_column] for sample in samples])
        if not isinstance(inputs, Mapping) or not {
            "input_ids",
            "attention_mask",
        }.issubset(inputs):
            raise ValueError("Tokenizer must return input_ids and attention_mask")
        targets = torch.tensor(
            [sample[config.label_column] for sample in samples], dtype=torch.long
        )
        if (targets < 0).any().item():
            raise ValueError(
                "Classification data needs labeled samples, not negative/missing class IDs"
            )
        return inputs, targets

    sampler = train_sampler(train)
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=collate,
            num_workers=0,
        ),
        DataLoader(
            validation, batch_size=batch_size, collate_fn=collate, num_workers=0
        ),
    )


def build_test_dataset(config: DatasetConfig, *, seed: int = 42) -> Dataset:
    """Load the held-out fraction of the labeled validation source split."""
    _, test = _build_dataset(config, seed=seed, _test=True)
    return test
