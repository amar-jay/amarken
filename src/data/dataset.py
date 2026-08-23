"""Streaming datasets for training and evaluation of models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any, Iterator, Literal

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from src.tokenization.tokenizer import AmarkenTokenizer, load_tokenizer
from src.data.chat_format import render_chat


DatasetSplit = Literal["train", "validation", "all"]


class AmarkenDataset(IterableDataset[dict[str, Any]]):
    """Stream synthetic JSONL records, selecting the split stored in each row.

    ``dataset`` may be the pretraining root or its ``shards`` directory. The
    dataset never materializes the corpus in memory. When passed to a multi-worker
    ``DataLoader``, shards are divided across workers to avoid duplicate samples.
    """

    def __init__(
        self,
        dataset: Path | str,
        split: DatasetSplit = "train",
        *,
        shuffle: bool = False,
        seed: int = 0,
        shuffle_buffer_size: int = 10_000,
    ) -> None:
        super().__init__()
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be 'train', 'validation', or 'all'")
        if shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be positive")
        dataset_path = Path(dataset)
        shards_path = dataset_path / "shards" if (dataset_path / "shards").is_dir() else dataset_path
        self.shards = tuple(sorted(shards_path.glob("*.jsonl")))
        if not self.shards:
            raise ValueError(f"no JSONL shards found in {shards_path}")
        self.split = split
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic shuffle epoch before constructing an iterator."""
        self.epoch = epoch

    def _records(self, shards: tuple[Path, ...]) -> Iterator[dict[str, Any]]:
        for shard in shards:
            with shard.open("r", encoding="utf-8", errors="strict") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"invalid JSON in {shard}:{line_number}") from error
                    if self.split == "all" or row.get("split") == self.split:
                        yield row

    @staticmethod
    def _buffered_shuffle(
        records: Iterator[dict[str, Any]], rng: random.Random, buffer_size: int
    ) -> Iterator[dict[str, Any]]:
        buffer: list[dict[str, Any]] = []
        for row in records:
            if len(buffer) < buffer_size:
                buffer.append(row)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = row
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        except RuntimeError:
            rank, world_size = 0, 1
        consumer_id = rank * worker_count + worker_id
        consumer_count = world_size * worker_count
        shards = self.shards[consumer_id::consumer_count]
        rng = random.Random(self.seed + self.epoch * 1_000_003 + consumer_id)
        if self.shuffle:
            shards = tuple(rng.sample(shards, len(shards)))
            yield from self._buffered_shuffle(
                self._records(shards), rng, self.shuffle_buffer_size
            )
            return
        yield from self._records(shards)


class AmarkenDataLoader(DataLoader):
    """DataLoader whose default collation preserves variable-length records."""

    def __init__(self, dataset: AmarkenDataset, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("collate_fn", lambda records: records)
        super().__init__(dataset, *args, **kwargs)


class PackedConversationDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Pack complete chats into blocks while isolating cross-chat attention.

    A conversation is never split across blocks because attention cannot cross a
    model forward-pass boundary. Records larger than the context fail explicitly.
    """

    def __init__(self, records: AmarkenDataset, tokenizer: AmarkenTokenizer,
                 sequence_length: int, *, token_budget: int | None = None) -> None:
        super().__init__()
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if token_budget is not None and token_budget < 1:
            raise ValueError("token_budget must be positive")
        self.records = records
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.token_budget = token_budget

    def set_epoch(self, epoch: int) -> None:
        self.records.set_epoch(epoch)

    def _encode(self, record: dict[str, Any]) -> tuple[list[int], list[int]]:
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"record {record.get('id', '<unknown>')!r} has no messages")
        rendered = render_chat(messages)
        if rendered is None or not rendered.assistant_spans:
            raise ValueError(f"record {record.get('id', '<unknown>')!r} has no assistant target")
        if any(role not in {"system", "user", "assistant"} for role in rendered.roles):
            raise ValueError("messages require a supported role and string content")

        # Tokenize the complete canonical string once. Byte-BPE can merge a
        # leading space with the first content character, so concatenating token
        # lists for prefix/body/suffix would not reproduce corpus tokenization.
        ids, offsets = self.tokenizer.encode_with_offsets(rendered.text)
        labels = [-100] * len(ids)
        for index, (start, end) in enumerate(offsets):
            if any(span.start <= start and end <= span.end for span in rendered.assistant_spans):
                labels[index] = ids[index]
        return ids, labels

    def _block(self, ids: list[int], labels: list[int], segments: list[int]) -> dict[str, torch.Tensor]:
        padding = self.sequence_length - len(ids)
        return {
            "input_ids": torch.tensor(ids + [self.tokenizer.pad_id()] * padding, dtype=torch.long),
            "labels": torch.tensor(labels + [-100] * padding, dtype=torch.long),
            "attention_mask": torch.tensor([True] * len(ids) + [False] * padding),
            "segment_ids": torch.tensor(segments + [-1] * padding, dtype=torch.long),
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        block_ids: list[int] = []
        block_labels: list[int] = []
        block_segments: list[int] = []
        emitted_tokens = 0
        local_budget = self.token_budget
        if local_budget is not None:
            worker = get_worker_info()
            worker_count = worker.num_workers if worker is not None else 1
            try:
                import torch.distributed as dist
                world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
            except RuntimeError:
                world_size = 1
            local_budget = math.ceil(local_budget / (worker_count * world_size))
        segment = 0
        for record in self.records:
            ids, labels = self._encode(record)
            if len(ids) > self.sequence_length:
                # Splitting here would orphan the continuation in the next block:
                # segment IDs isolate tokens only inside one block and cannot make
                # attention reach context stored in a previous forward pass.
                raise ValueError(
                    f"record {record.get('id', '<unknown>')!r} needs {len(ids)} tokens "
                    f"but sequence_length is {self.sequence_length}"
                )

            if block_ids and len(block_ids) + len(ids) > self.sequence_length:
                # Keep conversations atomic. Padding costs some compute, but every
                # supervised assistant token retains its system/user context.
                if any(label != -100 for label in block_labels[1:]):
                    yield self._block(block_ids, block_labels, block_segments)
                    emitted_tokens += self.sequence_length
                if local_budget is not None and emitted_tokens >= local_budget:
                    return
                block_ids, block_labels, block_segments = [], [], []

            block_ids.extend(ids)
            block_labels.extend(labels)
            block_segments.extend([segment] * len(ids))
            segment += 1
        if (block_ids and any(label != -100 for label in block_labels[1:])
                and (local_budget is None or emitted_tokens < local_budget)):
            yield self._block(block_ids, block_labels, block_segments)


def inspect_dataset(
    dataset_path: Path | str,
    tokenizer_path: Path | str,
    *,
    split: DatasetSplit = "train",
    sequence_length: int = 256,
    samples: int = 3,
    show_tokens: bool = False,
    shuffle: bool = False,
    seed: int = 2026,
    epoch: int = 0,
) -> dict[str, int]:
    """Print packed examples and assert the invariants expected by the trainer.

    This intentionally constructs ``PackedConversationDataset`` instead of
    duplicating its formatting logic, so the preview shows exactly what training
    receives. The returned counts also make the function convenient for tests.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    tokenizer = load_tokenizer(tokenizer_path)
    records = AmarkenDataset(
        dataset_path,
        split,
        shuffle=shuffle,
        seed=seed,
    )
    # Training changes this value at every epoch. Setting it here lets the CLI
    # reproduce any epoch exactly instead of using nondeterministic randomness.
    records.set_epoch(epoch)
    packed = PackedConversationDataset(
        records,
        tokenizer,
        sequence_length,
    )
    totals = {"blocks": 0, "tokens": 0, "supervised_tokens": 0, "segments": 0}

    for block_index, block in enumerate(packed, start=1):
        valid = block["attention_mask"].bool()
        ignored = block["labels"].eq(-100)
        supervised = valid & ~ignored

        # Fail loudly here instead of discovering malformed packing after a long
        # run. Labels store the token to predict at that position; the model does
        # the standard one-token shift when calculating causal cross-entropy.
        if not torch.equal(block["labels"][supervised], block["input_ids"][supervised]):
            raise RuntimeError("supervised labels do not match their input tokens")
        if not bool(ignored[~valid].all()):
            raise RuntimeError("padding positions must have ignored labels")
        if not bool(block["segment_ids"][~valid].eq(-1).all()):
            raise RuntimeError("padding positions must use segment id -1")
        if not bool(supervised[1:].any()):
            raise RuntimeError("packed block has no causal training targets")

        valid_ids = block["input_ids"][valid].tolist()
        supervised_ids = block["labels"][supervised].tolist()
        segment_ids = block["segment_ids"][valid]
        unique_segments = segment_ids.unique_consecutive().tolist()
        totals["blocks"] += 1
        totals["tokens"] += len(valid_ids)
        totals["supervised_tokens"] += len(supervised_ids)
        totals["segments"] += len(unique_segments)

        print(f"\n=== packed block {block_index} ===")
        print(
            f"valid={len(valid_ids)}/{sequence_length}  "
            f"supervised={len(supervised_ids)}  segments={len(unique_segments)}"
        )
        print("\nINPUT (everything visible to the model):")
        print(tokenizer.decode(valid_ids))
        print("\nTARGET SPANS (only contiguous assistant-token losses):")
        for segment_id in unique_segments:
            segment_mask = valid & block["segment_ids"].eq(segment_id)
            target_mask = segment_mask & ~ignored
            positions = target_mask.nonzero(as_tuple=False).flatten().tolist()
            if not positions:
                print(f"[{segment_id}] <no target>")
                continue
            # One chat can contain several assistant turns. The user turn between
            # them remains in input_ids as context but has -100 labels; printing
            # each contiguous run prevents the preview from hiding that boundary.
            span_start = positions[0]
            previous = positions[0]
            for position in positions[1:] + [None]:
                if position is not None and position == previous + 1:
                    previous = position
                    continue
                target_ids = block["labels"][span_start : previous + 1].tolist()
                print(
                    f"[segment {segment_id}, positions {span_start}:{previous + 1}] "
                    f"{tokenizer.decode(target_ids)}"
                )
                if position is not None:
                    span_start = position
                    previous = position
        print("\nSEGMENTS (attention is isolated between these):")
        for segment_id in unique_segments:
            mask = valid & block["segment_ids"].eq(segment_id)
            print(f"[{segment_id}] {tokenizer.decode(block['input_ids'][mask].tolist())}")

        if show_tokens:
            print("\nTOKENS (S means supervised):")
            for position in valid.nonzero(as_tuple=False).flatten().tolist():
                token_id = int(block["input_ids"][position])
                flag = "S" if int(block["labels"][position]) != -100 else "-"
                segment_id = int(block["segment_ids"][position])
                print(
                    f"{position:4d}  seg={segment_id:4d}  {flag}  "
                    f"id={token_id:5d}  {tokenizer.piece(token_id)!r}"
                )
        if block_index >= samples:
            break

    if totals["blocks"] == 0:
        raise RuntimeError(f"no supervised packed blocks found for split {split!r}")
    print(
        "\n=== inspection summary ===\n"
        f"blocks={totals['blocks']}  tokens={totals['tokens']}  "
        f"supervised_tokens={totals['supervised_tokens']}  segments={totals['segments']}"
    )
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the exact tokenized and packed blocks used for training."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/synthetic/pretraining/shards"),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizers/v3/tiktoken-tr-bpe-12k.json"),
    )
    parser.add_argument("--split", choices=("train", "validation", "all"), default="train")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="inspect the deterministic shuffled training order",
    )
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="print every token id, piece, segment, and supervision flag",
    )
    args = parser.parse_args()
    inspect_dataset(
        args.dataset,
        args.tokenizer,
        split=args.split,
        sequence_length=args.sequence_length,
        samples=args.samples,
        show_tokens=args.show_tokens,
        shuffle=args.shuffle,
        seed=args.seed,
        epoch=args.epoch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
