"""Numeric example: build, restore and predict with the same public interface as base."""

from pathlib import Path
from typing import Any

import torch
from torch import nn

from amarken.constants import (
    EXAMPLE_HIDDEN_SIZE,
    EXAMPLE_INPUT_SIZE,
    EXAMPLE_NUM_CLASSES,
)
from amarken.utils import load_checkpoint_weights


def build_model(*, device: str = "cpu") -> nn.Module:
    return nn.Sequential(
        nn.Linear(EXAMPLE_INPUT_SIZE, EXAMPLE_HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(EXAMPLE_HIDDEN_SIZE, EXAMPLE_NUM_CLASSES),
    ).to(device)


def load_model(checkpoint: str | Path, *, device: str = "cpu") -> nn.Module:
    model = build_model()
    model.load_state_dict(load_checkpoint_weights(checkpoint), strict=True)
    return model.to(device).eval()


def predict(model: nn.Module, inputs: Any) -> dict:
    """Predict with an already-loaded model; inputs are numeric features."""
    try:
        features = torch.tensor(inputs, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Input must be a numeric JSON array") from error
    if features.ndim == 1:
        features = features.unsqueeze(0)
    if (
        features.ndim != 2
        or features.shape[0] == 0
        or features.shape[1] != EXAMPLE_INPUT_SIZE
    ):
        raise ValueError(
            f"Expected one sample of {EXAMPLE_INPUT_SIZE} features or a nonempty batch with shape [N, {EXAMPLE_INPUT_SIZE}]"
        )
    if not torch.isfinite(features).all().item():
        raise ValueError("Input features must be finite")
    parameter = next(model.parameters())
    model.eval()
    with torch.inference_mode():
        probabilities = model(
            features.to(device=parameter.device, dtype=parameter.dtype)
        ).softmax(-1)
    return {
        "predictions": probabilities.argmax(-1).cpu().tolist(),
        "probabilities": probabilities.cpu().tolist(),
    }
