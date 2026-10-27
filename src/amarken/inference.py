"""Checkpoint inference using the implementation selected in model.py."""

from pathlib import Path
from typing import Any

from amarken.model import load_model, predict


def run_inference(
    checkpoint: str | Path,
    inputs: Any,
    *,
    model_id: str | None = None,
    device: str = "cpu",
    precision: str = "float32",
) -> dict:
    model = load_model(
        checkpoint, model_id=model_id, device=device, precision=precision
    )
    return predict(model, inputs)
