"""Small paired LJSpeech example; keeps audio encoded until model preprocessing."""

import sys
from collections.abc import Callable
from functools import partial
from hashlib import sha256

import pyarrow.parquet as pq
from datasets import Dataset
from huggingface_hub import hf_hub_download, list_repo_files
from torch.utils.data import DataLoader

from amarken.constants import DATASET_REVISIONS, HUB_CACHE
from amarken.datasets.config import DatasetConfig
from amarken.utils.distributed import primary, train_sampler


def _source(config, split):
    if primary():
        print(
            f"Loading TTS dataset {config.id} ({split})...", file=sys.stderr, flush=True
        )
    revision = DATASET_REVISIONS.get(config.id)
    prefix = f"{config.config or 'data'}/{split}-"
    files = sorted(
        name
        for name in list_repo_files(config.id, repo_type="dataset", revision=revision)
        if name.startswith(prefix) and name.endswith(".parquet")
    )
    if not files:
        raise ValueError(f"TTS demo needs a Parquet shard matching {prefix}*.parquet")
    for filename in files:
        if primary():
            print(f"Caching TTS shard {filename}...", file=sys.stderr, flush=True)
        path = hf_hub_download(
            config.id,
            filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=HUB_CACHE,
        )
        # Local reads avoid remote Arrow scanner stalls. This batch size controls
        # I/O memory only; it never caps the requested number of recordings.
        with pq.ParquetFile(path) as source:
            for batch in source.iter_batches(
                batch_size=64,
                columns=[config.text_column, config.label_column],
                use_threads=False,
            ):
                for row in batch.to_pylist():
                    yield {
                        "text": row[config.text_column],
                        "audio": row[config.label_column],
                    }


def _partition(row):
    # Stable across requested sizes and sampling seeds; identical transcripts
    # always stay together rather than leaking between evaluation and training.
    return int.from_bytes(sha256(row["text"].encode()).digest()[:8], "big") / 2**64


def _collect(source, sizes, assign):
    buckets = [[] for _ in sizes]
    for row in source:
        index = assign(row)
        if len(buckets[index]) < sizes[index]:
            buckets[index].append(row)
        if all(
            len(bucket) == size for bucket, size in zip(buckets, sizes, strict=True)
        ):
            return buckets
    actual = [len(bucket) for bucket in buckets]
    raise ValueError(
        f"TTS dataset cannot supply requested split sizes {sizes}; available {actual}"
    )


def _build_dataset(config, *, seed=42, test=False):
    sizes = [config.train_size, config.val_size, config.test_size]
    if config.train_split == config.validation_split:

        def assign(row):
            value = _partition(row)
            if value < 0.8:
                return 0
            return 1 if value < 1 - 0.2 * config.test_fraction else 2

        train, val, heldout = _collect(
            _source(config, config.train_split), sizes, assign
        )
    else:
        train = _collect(_source(config, config.train_split), sizes[:1], lambda row: 0)[
            0
        ]
        val, heldout = _collect(
            _source(config, config.validation_split),
            sizes[1:],
            lambda row: int(_partition(row) >= 1 - config.test_fraction),
        )
    return (
        Dataset.from_list(train).shuffle(seed=seed),
        Dataset.from_list(heldout if test else val).shuffle(seed=seed),
    )


def build_dataset(config: DatasetConfig, *, seed: int = 42):
    return _build_dataset(config, seed=seed)


def build_test_dataset(config: DatasetConfig, *, seed: int = 42):
    return _build_dataset(config, seed=seed, test=True)[1]


def build_dataloaders(
    config: DatasetConfig,
    train,
    validation,
    *,
    batch_size: int,
    tokenize: Callable | None = None,
):
    if not callable(tokenize):
        raise TypeError("TTS data requires model.tokenize")
    loader = partial(
        DataLoader, batch_size=batch_size, collate_fn=tokenize, num_workers=0
    )
    sampler = train_sampler(train)
    return (
        loader(train, sampler=sampler, shuffle=sampler is None),
        loader(validation),
    )
