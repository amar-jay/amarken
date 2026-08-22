"""Deterministic packed-token data structures for causal LM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class TokenizedExample:
    """One conversation/document and the exact tokens eligible for LM loss."""

    input_ids: tuple[int, ...]
    assistant_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.input_ids or len(self.input_ids) != len(self.assistant_mask):
            raise ValueError("input_ids and nonempty assistant_mask must have equal length")


@dataclass(frozen=True)
class PackedBlock:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[bool, ...]


class PackedSequenceDataset(Sequence[PackedBlock]):
    """Greedily concatenate examples into fixed blocks without wasting context.

    Packing is performed once in caller-provided order. Shuffling later operates
    on complete blocks, which makes resume state a compact epoch/block cursor and
    prevents worker scheduling from changing example order.
    """

    def __init__(
        self,
        examples: Iterable[TokenizedExample],
        sequence_length: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> None:
        if sequence_length < 2 or min(eos_token_id, pad_token_id) < 0:
            raise ValueError("sequence_length must be >=2 and token IDs nonnegative")
        tokens: list[int] = []
        labels: list[int] = []
        blocks: list[PackedBlock] = []

        def emit() -> None:
            if not tokens:
                return
            valid = len(tokens)
            padding = sequence_length - valid
            block_labels = list(labels)
            # Position zero has no in-block predecessor. This also handles a long
            # document sliced across blocks, where the preceding token is absent.
            block_labels[0] = -100
            blocks.append(PackedBlock(
                input_ids=tuple(tokens + [pad_token_id] * padding),
                labels=tuple(block_labels + [-100] * padding),
                attention_mask=tuple([True] * valid + [False] * padding),
            ))
            tokens.clear()
            labels.clear()

        for example in examples:
            # Long records are sliced rather than discarded. Each slice preserves
            # its original assistant eligibility; no synthetic target is introduced.
            stream_tokens = list(example.input_ids) + [eos_token_id]
            # EOS closes a record but is not trained by default because the source
            # mask cannot tell whether the preceding assistant turn ended the sample.
            stream_labels = [token if mask else -100 for token, mask in zip(example.input_ids, example.assistant_mask)] + [-100]
            # A packed record's first token would otherwise be predicted from the
            # preceding record's EOS/context. Mask that artificial cross-document
            # transition even when the record starts directly with an assistant.
            stream_labels[0] = -100
            offset = 0
            while offset < len(stream_tokens):
                take = min(sequence_length - len(tokens), len(stream_tokens) - offset)
                tokens.extend(stream_tokens[offset:offset + take])
                labels.extend(stream_labels[offset:offset + take])
                offset += take
                if len(tokens) == sequence_length:
                    emit()
        emit()
        if not blocks:
            raise ValueError("at least one tokenized example is required")
        self.blocks = tuple(blocks)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> PackedBlock:
        return self.blocks[index]

    def batch(self, indices: Sequence[int], device: torch.device) -> dict[str, Tensor]:
        if not indices:
            raise ValueError("batch indices cannot be empty")
        selected = [self.blocks[index] for index in indices]
        return {
            "input_ids": torch.tensor([block.input_ids for block in selected], dtype=torch.long, device=device),
            "labels": torch.tensor([block.labels for block in selected], dtype=torch.long, device=device),
            "attention_mask": torch.tensor([block.attention_mask for block in selected], dtype=torch.bool, device=device),
        }
