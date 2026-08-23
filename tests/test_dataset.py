import json
import re
from pathlib import Path

import pytest

from src.data.dataset import AmarkenDataLoader, AmarkenDataset, PackedConversationDataset


def _write_shard(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_streams_existing_record_splits_from_dataset_root(tmp_path: Path):
    shards = tmp_path / "shards"
    shards.mkdir()
    _write_shard(
        shards / "shard-000000.jsonl",
        [
            {"id": "train-1", "split": "train"},
            {"id": "validation-1", "split": "validation"},
        ],
    )
    _write_shard(
        shards / "translations-000000.jsonl",
        [{"id": "train-2", "split": "train"}],
    )

    assert [row["id"] for row in AmarkenDataset(tmp_path, "train")] == [
        "train-1",
        "train-2",
    ]
    assert [row["id"] for row in AmarkenDataset(shards, "validation")] == [
        "validation-1"
    ]


def test_shuffle_is_reproducible_and_epoch_sensitive(tmp_path: Path):
    _write_shard(
        tmp_path / "shard-000000.jsonl",
        [{"id": str(index), "split": "train"} for index in range(20)],
    )
    dataset = AmarkenDataset(
        tmp_path, shuffle=True, seed=17, shuffle_buffer_size=4
    )
    first = [row["id"] for row in dataset]
    second = [row["id"] for row in dataset]
    dataset.set_epoch(1)
    third = [row["id"] for row in dataset]

    assert first == second
    assert set(first) == {str(index) for index in range(20)}
    assert first != third


def test_rejects_invalid_split_or_empty_source(tmp_path: Path):
    with pytest.raises(ValueError, match="no JSONL"):
        AmarkenDataset(tmp_path)
    _write_shard(tmp_path / "shard-000000.jsonl", [])
    with pytest.raises(ValueError, match="split must"):
        AmarkenDataset(tmp_path, "test")  # type: ignore[arg-type]


def test_dataloader_batches_full_variable_length_records(tmp_path: Path):
    _write_shard(
        tmp_path / "shard-000000.jsonl",
        [
            {"id": "one", "split": "train", "messages": [{"content": "a"}]},
            {
                "id": "two",
                "split": "train",
                "messages": [{"content": "b"}, {"content": "c"}],
            },
        ],
    )

    batch = next(iter(AmarkenDataLoader(AmarkenDataset(tmp_path), batch_size=2)))
    assert [record["id"] for record in batch] == ["one", "two"]
    assert len(batch[1]["messages"]) == 2


class _TinyTokenizer:
    def __init__(self):
        self.next_id = 10

    def encode(self, text: str) -> list[int]:
        result = list(range(self.next_id, self.next_id + max(1, len(text.split()))))
        self.next_id += len(result)
        return result

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        offsets = [match.span() for match in re.finditer(r"\S+", text)]
        return self.encode(text), offsets

    def pad_id(self) -> int: return 3


def test_packed_conversations_supervise_only_assistant_and_pad(tmp_path: Path):
    _write_shard(tmp_path / "shard-000000.jsonl", [{
        "id": "chat", "split": "train", "messages": [
            {"role": "user", "content": "private prompt"},
            {"role": "assistant", "content": "visible answer"},
        ]}])
    dataset = PackedConversationDataset(
        AmarkenDataset(tmp_path), _TinyTokenizer(), 16  # type: ignore[arg-type]
    )
    block = next(iter(dataset))

    assert block["input_ids"].shape == (16,)
    assert block["labels"].ne(-100).any()
    assert (block["labels"][block["attention_mask"].logical_not()] == -100).all()
    assert (block["segment_ids"][block["attention_mask"].logical_not()] == -1).all()


def test_packing_never_splits_a_conversation_across_blocks(tmp_path: Path):
    _write_shard(tmp_path / "shard-000000.jsonl", [
        {"id": str(index), "split": "train", "messages": [
            {"role": "user", "content": "one two three four"},
            {"role": "assistant", "content": "five six seven eight"},
        ]}
        for index in range(2)
    ])
    blocks = list(PackedConversationDataset(
        AmarkenDataset(tmp_path), _TinyTokenizer(), 16  # type: ignore[arg-type]
    ))

    assert len(blocks) == 2
    assert [block["attention_mask"].sum().item() for block in blocks] == [12, 12]
    assert [block["segment_ids"][block["attention_mask"]].unique().numel()
            for block in blocks] == [1, 1]


def test_conversation_larger_than_context_fails_instead_of_orphaning_targets(tmp_path: Path):
    _write_shard(tmp_path / "shard-000000.jsonl", [{
        "id": "too-long", "split": "train", "messages": [
            {"role": "user", "content": "one two three four"},
            {"role": "assistant", "content": "five six seven eight"},
        ]}])
    dataset = PackedConversationDataset(
        AmarkenDataset(tmp_path), _TinyTokenizer(), 8  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="too-long.*needs 12 tokens"):
        next(iter(dataset))
