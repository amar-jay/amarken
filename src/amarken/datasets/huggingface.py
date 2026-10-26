"""Hugging Face text-classification data for modules.classification."""

from collections.abc import Callable, Mapping

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from amarken.constants import (
    DATA_CACHE,
    HF_DATASET_CONFIG,
    HF_DATASET_ID,
    HF_DATASET_LABEL_COLUMN,
    HF_DATASET_REVISION,
    HF_DATASET_TEXT_COLUMN,
    HF_DATASET_TRAIN_SPLIT,
    HF_DATASET_VALIDATION_SPLIT,
)
from amarken.distributed import train_sampler


def _build_dataset(
    *,
    seed: int = 42,
    train_size: int = 128,
    val_size: int = 32,
    _test: bool = False,
) -> tuple[Dataset, Dataset]:
    train, validation = load_dataset(
        HF_DATASET_ID,
        name=HF_DATASET_CONFIG,
        revision=HF_DATASET_REVISION,
        split=[HF_DATASET_TRAIN_SPLIT, HF_DATASET_VALIDATION_SPLIT],
        cache_dir=DATA_CACHE,
    )

    # A fixed partition seed keeps validation/test membership independent of
    # the run's sampling seed. SST-2's public test labels are unavailable.
    if len(validation) < 2:
        raise ValueError("Need at least two labeled rows for validation and test")
    validation = validation.shuffle(seed=0)
    boundary = len(validation) // 2
    validation = validation.select(
        range(boundary, len(validation)) if _test else range(boundary)
    )

    def sample(dataset, size):
        required = {HF_DATASET_TEXT_COLUMN, HF_DATASET_LABEL_COLUMN}
        if not required.issubset(dataset.column_names):
            raise ValueError(f"Dataset requires columns {sorted(required)}")
        if len(dataset) == 0:
            raise ValueError("Dataset splits must not be empty")
        if size is not None:
            if type(size) is not int or size < 1:
                raise ValueError(
                    "Dataset sample limits must be positive integers or None"
                )
            dataset = dataset.shuffle(seed=seed).select(range(min(size, len(dataset))))
        return dataset

    return sample(train, train_size), sample(validation, val_size)


def build_dataset(
    *, seed: int = 42, train_size: int = 128, val_size: int = 32
) -> tuple[Dataset, Dataset]:
    return _build_dataset(seed=seed, train_size=train_size, val_size=val_size)


def build_dataloaders(
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
        inputs = tokenize([sample[HF_DATASET_TEXT_COLUMN] for sample in samples])
        if not isinstance(inputs, Mapping) or not {
            "input_ids",
            "attention_mask",
        }.issubset(inputs):
            raise ValueError("Tokenizer must return input_ids and attention_mask")
        targets = torch.tensor(
            [sample[HF_DATASET_LABEL_COLUMN] for sample in samples], dtype=torch.long
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


def build_test_dataset(*, seed: int = 42, test_size: int = 32) -> Dataset:
    """Load the held-out half of the labeled validation source split."""
    _, test = _build_dataset(seed=seed, train_size=1, val_size=test_size, _test=True)
    return test
