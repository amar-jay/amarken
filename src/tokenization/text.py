"""Text sanitation shared by tokenizer training and visualization."""

from __future__ import annotations

import re

from ftfy import fix_encoding
from ftfy.badness import badness


_RESIDUAL_MOJIBAKE = re.compile(r"[\u0080-\u009fÃÄÅ]")


def repair_text_encoding(text: str) -> tuple[str | None, str]:
    """Repair recoverable mojibake and reject suspicious residual text."""
    original_badness = badness(text)
    repaired = fix_encoding(text)
    if repaired != text and badness(repaired) < original_badness:
        text = repaired
        status = "repaired"
    else:
        status = "unchanged"
    if _RESIDUAL_MOJIBAKE.search(text):
        return None, "rejected_residual_mojibake"
    return text, status
