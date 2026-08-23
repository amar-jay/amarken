"""Canonical serialization for Amarken chat records.

Every consumer of ``messages`` must use this module: tokenizer-corpus creation,
runtime training, visualization, and eventual inference prompting. Keeping the
exact punctuation and whitespace here is essential for byte-level BPE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


ROLE_TAGS = {
    "system": "<|system|>",
    "developer": "<|developer|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
}


@dataclass(frozen=True)
class AssistantSpan:
    """Character range of one assistant target in a rendered chat string."""

    start: int
    end: int


@dataclass(frozen=True)
class RenderedChat:
    text: str
    roles: tuple[str, ...]
    assistant_spans: tuple[AssistantSpan, ...]


def render_chat(messages: Sequence[dict[str, Any]]) -> RenderedChat | None:
    """Render stored messages with the exact template used for tokenizer corpus.

    Assistant spans start after ``<|assistant|>:`` and include its leading space,
    content, trailing ``" \\n"``, and ``<|end|>``. This trains the complete
    assistant turn while leaving its role marker and every non-assistant turn as
    context-only tokens.
    """
    rendered: list[str] = []
    roles: list[str] = []
    spans: list[AssistantSpan] = []
    cursor = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if content is None:
            continue
        content = str(content).strip()
        if not content:
            continue
        role = str(message.get("role", "unknown")).lower()
        tag = ROLE_TAGS.get(role, f"<|{role}|>")
        prefix = f"{tag}:"
        turn = f"{prefix} {content} \n<|end|>"
        if rendered:
            cursor += 1  # The newline inserted between consecutive turns.
        turn_start = cursor
        rendered.append(turn)
        roles.append(role)
        if role == "assistant":
            spans.append(AssistantSpan(turn_start + len(prefix), turn_start + len(turn)))
        cursor += len(turn)
    if not rendered:
        return None
    return RenderedChat("\n".join(rendered), tuple(roles), tuple(spans))
