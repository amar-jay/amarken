"""Checkpoint inference using the implementation selected in model.py."""

from pathlib import Path
from typing import Any

from amarken.model import load_model, predict


def run_inference(checkpoint: str | Path, inputs: Any, *, device: str = "cpu") -> dict:
    model = load_model(checkpoint, device=device)
    return predict(model, inputs)
