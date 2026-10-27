"""Plain-text next-token data; reuse the HF source splitting and sampling hooks.

The starter config uses SST-2 sentences (class labels are ignored). Replace this
source for a real language corpus or an instruction/response SFT objective.
"""

from dataclasses import replace

from torch.utils.data import DataLoader

from amarken.datasets.classification import (
    build_dataset as build_text_dataset,
)
from amarken.datasets.classification import (
    build_test_dataset as build_text_test_dataset,
)
from amarken.utils.distributed import train_sampler

__all__ = ["build_dataloaders", "build_dataset", "build_test_dataset"]


def build_dataset(config, *, seed=42):
    # Reuse source partitioning without requiring a classification label column.
    return build_text_dataset(
        replace(config, label_column=config.text_column), seed=seed
    )


def build_test_dataset(config, *, seed=42):
    return build_text_test_dataset(
        replace(config, label_column=config.text_column), seed=seed
    )


def build_dataloaders(config, train, validation, *, batch_size, tokenize=None):
    if not callable(tokenize):
        raise TypeError("Causal data requires model.tokenize")

    def collate(samples):
        inputs = tokenize([row[config.text_column] for row in samples])
        labels = inputs["input_ids"].clone()
        labels.masked_fill_(inputs["attention_mask"] == 0, -100)
        return inputs, labels

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
