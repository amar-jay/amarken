"""Small, dependency-free helpers for file-backed experiment artifacts."""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path


def sha256_file(path: Path | str, *, block_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 digest without loading the whole file into memory."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()

def write_report(report: dict, report_path: Path) -> None:
    """Write a JSON report to disk atomically, creating parent directories as needed."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, report_path)