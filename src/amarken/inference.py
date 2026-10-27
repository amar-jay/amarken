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
    fine_tuning=None,
) -> dict:
    options = {"fine_tuning": fine_tuning} if fine_tuning is not None else {}
    model = load_model(
        checkpoint, model_id=model_id, device=device, precision=precision, **options
    )
    return predict(model, inputs)
