"""Load training checkpoint weights and run the synthetic classification task."""

from pathlib import Path
from typing import Any

import torch

from amarken.model import load_model


def predict_example(
    checkpoint: str | Path, inputs: Any, *, device: str = "cpu"
) -> dict:
    """Predict classes for one eight-feature sample or a batch of samples."""
    try:
        features = torch.tensor(inputs, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Input must be a numeric JSON array") from error
    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != 8:
        raise ValueError(
            "Expected one sample of 8 features or a nonempty batch with shape [N, 8]"
        )
    if not torch.isfinite(features).all().item():
        raise ValueError("Input features must be finite")
    model = load_model(checkpoint, device=device)
    with torch.inference_mode():
        logits = model(features.to(device))
        probabilities = logits.softmax(dim=-1)
    return {
        "predictions": probabilities.argmax(dim=-1).cpu().tolist(),
        "probabilities": probabilities.cpu().tolist(),
    }
