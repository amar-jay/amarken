from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from amarken.modules.example import build_example_model


def load_model(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    model_factory: Callable[[Mapping[str, Any]], torch.nn.Module] | None = None,
) -> torch.nn.Module:
    """Rebuild a model from saved config and weights, without training setup.

    Custom factories accept the checkpoint's config dictionary and return the
    matching architecture. The built-in architecture is synthetic-mlp only.
    Optimizer, scheduler and RNG state are not restored for inference.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise ValueError("Expected an amarken training checkpoint (format_version=1)")
    config = state.get("config")
    if not isinstance(config, dict) or "model" not in state:
        raise ValueError("Checkpoint must contain config and model weights")
    if model_factory is None:
        if config.get("model_id") != "synthetic-mlp":
            raise ValueError(
                "This model requires a custom model_factory; the CLI supports synthetic-mlp"
            )
        model = build_example_model()
    else:
        model = model_factory(config)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()
