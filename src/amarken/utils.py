"""Shared training setup and interactive helpers."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import questionary

from amarken import logger as logger_module

if TYPE_CHECKING:
    from amarken.trainer import Trainer, TrainingConfig


def prompt_run_name(config: "TrainingConfig", *, resume: bool = False) -> str | None:
    """Ask only for fresh isolated runs; blank/noninteractive uses UUID."""
    if config.checkpoints.overwrite or resume or not sys.stdin.isatty():
        return None

    def validate(value):
        name = value.strip()
        if name and (Path(name).name != name or name in {".", ".."}):
            return "Use a single directory name without path separators."
        return True

    answer = questionary.text(
        "Run name (leave blank for UUID):", default="", validate=validate
    ).ask()
    if answer is None:
        raise SystemExit(130)
    return answer.strip() or None


def prepare_run(trainer: "Trainer") -> None:
    """Resolve a name and reserve its directory once, before fit starts work."""
    if trainer._directory_prepared:
        return
    if trainer.run_label is None:
        trainer.run_label = prompt_run_name(trainer.config, resume=trainer._resumed)
    if not trainer.config.checkpoints.overwrite:
        directory = logger_module.CHECKPOINT_DIR / trainer.checkpoint_prefix
        # Only an explicit resume may reuse an isolated run's directory.
        directory.mkdir(parents=True, exist_ok=trainer._resumed)
    trainer._directory_prepared = True


def load_checkpoint_weights(checkpoint: str | Path) -> dict:
    """Read model weights from an amarken training checkpoint, on CPU."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise ValueError("Expected an amarken training checkpoint (format_version=1)")
    if not isinstance(state.get("model"), dict):
        raise TypeError("Checkpoint must contain model weights")
    return state["model"]


def prepare_training_precision(model, precision: str) -> None:
    """Keep trainable weights/buffers in FP32; frozen subtrees may use AMP dtype.

    Run before moving to the accelerator, wrapping, or constructing the optimizer.
    Parameter objects are retained for callers supplying an existing optimizer.
    """
    import torch

    if not any(parameter.requires_grad for parameter in model.parameters()):
        model.to(dtype=getattr(torch, precision))
        return
    for parameter in model.parameters(recurse=False):
        if parameter.is_floating_point():
            dtype = (
                torch.float32 if parameter.requires_grad else getattr(torch, precision)
            )
            parameter.data = parameter.data.to(dtype=dtype)
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.to(dtype=dtype)
    for name, buffer in model.named_buffers(recurse=False):
        if buffer.is_floating_point():
            setattr(model, name, buffer.float())
    for child in model.children():
        prepare_training_precision(child, precision)
