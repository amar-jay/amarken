"""Synthetic addition with easy SFT and harder RL variants, and fixed splits."""

import hashlib
import random
from dataclasses import replace

import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from amarken.constants import (
    ARITHMETIC_HARD_OPERANDS,
    ARITHMETIC_HARD_POOL_SIZE,
    ARITHMETIC_MAX_OPERAND,
    CAUSAL_MAX_LENGTH,
)
from amarken.datasets.sft import collate_samples
from amarken.utils.distributed import train_sampler


def _operands(dataset_id):
    if dataset_id == "arithmetic":
        return (
            (a, b)
            for a in range(ARITHMETIC_MAX_OPERAND + 1)
            for b in range(a, ARITHMETIC_MAX_OPERAND + 1)
        )
    if dataset_id != "arithmetic-hard":
        raise ValueError("Arithmetic dataset.id must be arithmetic or arithmetic-hard")
    # A bounded, seed-independent source keeps held-out membership stable when
    # the run seed or requested sizes change. Canonical pairs exclude reversals.
    rng = random.Random(0)
    pairs = {}
    while len(pairs) < ARITHMETIC_HARD_POOL_SIZE:
        pair = tuple(sorted(rng.randint(*ARITHMETIC_HARD_OPERANDS) for _ in range(2)))
        pairs[pair] = None
    return pairs


def _build_dataset(config, *, seed):
    partitions = [[], [], []]
    for a, b in _operands(config.id):
        key = (
            int.from_bytes(hashlib.sha256(f"{a}+{b}".encode()).digest()[:8], "big")
            / 2**64
        )
        index = 0 if key < 0.8 else (1 if key < 1 - 0.2 * config.test_fraction else 2)
        partitions[index].append(
            {
                "prompt": f"What is {a} + {b}? Reply with only the integer answer.",
                "answer": a + b,
            }
        )
    result = []
    for index, (rows, size) in enumerate(
        zip(
            partitions,
            (config.train_size, config.val_size, config.test_size),
            strict=True,
        )
    ):
        if size > len(rows):
            raise ValueError(
                f"Arithmetic partition {index} has {len(rows)} rows; requested {size}"
            )
        random.Random(seed + index).shuffle(rows)
        result.append(Dataset.from_list(rows[:size]))
    return result


def build_dataset(config, *, seed=42):
    train, validation, _ = _build_dataset(config, seed=seed)
    return train, validation


def build_test_dataset(config, *, seed=42):
    return _build_dataset(config, seed=seed)[2]


def _generation_batch(rows):
    return {
        "prompts": [row["prompt"] for row in rows],
        "answers": torch.tensor([row["answer"] for row in rows]),
        "batch_size": len(rows),
    }


def build_rl_dataset(
    config, *, batch_size, tokenize, prepare_prompts, max_new_tokens, seed=42
):
    """Return formatted TRL records and an evaluation-only generation loader."""
    model = getattr(tokenize, "__self__", None)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise TypeError("Arithmetic RL requires model.tokenize with a tokenizer")
    train, validation = build_dataset(config, seed=seed)
    prompts = prepare_prompts(tokenizer, train["prompt"])
    context = min(CAUSAL_MAX_LENGTH, model.model.config.max_position_embeddings)
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if not tokens or len(tokens) + max_new_tokens > context:
            raise ValueError(
                "Arithmetic RL prompt plus completion exceeds the context limit"
            )
    records = train.remove_columns("prompt").add_column("prompt", prompts)
    evaluation = DataLoader(
        validation, batch_size=batch_size, collate_fn=_generation_batch, num_workers=0
    )
    return records, evaluation


def build_dataloaders(config, train, validation, *, batch_size, tokenize=None):
    tokenizer = getattr(getattr(tokenize, "__self__", None), "tokenizer", None)
    settings = replace(
        config, text_column="prompt", label_column="response", input_column=None
    )

    def collate(rows):
        batch = _generation_batch(rows)
        if tokenizer is not None:
            samples = [
                {"prompt": row["prompt"], "response": str(row["answer"])}
                for row in rows
            ]
            batch["inputs"], batch["labels"] = collate_samples(
                tokenizer, samples, settings
            )
        return batch

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
