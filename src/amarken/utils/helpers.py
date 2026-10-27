"""Shared training setup and interactive helpers."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from amarken.utils import logger as logger_module

if TYPE_CHECKING:
    from amarken.config import Config
    from amarken.trainer import Trainer


def prompt_run_name(config: "Config", *, resume: bool = False) -> str | None:
    """Ask only for fresh isolated runs; blank/noninteractive uses UUID."""
    if config.checkpoints.overwrite or resume or not sys.stdin.isatty():
        return None

    import questionary

    def validate(value):
        name = value.strip()
        if name and (Path(name).name != name or name in {".", ".."}):
            return "Use a single directory name without path separators."
        if name:
            directory = logger_module.CHECKPOINT_DIR / config.run_name / name
            if directory.exists() or directory.is_symlink():
                return f"Run '{name}' already exists. Choose another name, or use --resume to continue it."
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
        while True:
            directory = logger_module.CHECKPOINT_DIR / trainer.checkpoint_prefix
            directory.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Atomic reservation also catches collisions after prompt validation.
                directory.mkdir(exist_ok=trainer._resumed)
                break
            except FileExistsError:
                if trainer._resumed or not (
                    directory.exists() or directory.is_symlink()
                ):
                    raise
                message = (
                    f"Run '{trainer.run_label or trainer.run_id}' already exists. "
                    "Choose another name, or use --resume with its checkpoint to continue it."
                )
                if sys.stdin.isatty():
                    print(message, file=sys.stderr)
                    trainer.run_label = prompt_run_name(trainer.config)
                    if trainer.run_label is None:
                        trainer.run_id = uuid4().hex
                elif trainer.run_label is None:
                    trainer.run_id = uuid4().hex
                else:
                    raise FileExistsError(message) from None
    trainer._directory_prepared = True


def load_checkpoint_weights(checkpoint: str | Path) -> dict:
    """Read model weights from an amarken training checkpoint, on CPU."""
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
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


def seed_everything(seed: int) -> None:
    import random

    import torch

    """Call before constructing models and dataloaders."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def format_validation_table(result) -> str:
    """Format an evaluation result dictionary or JSON object for the console."""
    import json
    from collections.abc import Mapping

    if isinstance(result, str):
        result = json.loads(result)
    if not isinstance(result, Mapping):
        raise TypeError("Evaluation results must be a dictionary or JSON object")

    def cell(value):
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, list):
            return "[" + ", ".join(cell(item) for item in value) + "]"
        if isinstance(value, (dict, bool)) or value is None:
            return json.dumps(value, ensure_ascii=True)
        return str(value).replace("\n", " ").replace("\r", " ")

    def table(headers, rows):
        rows = [headers, *rows]
        widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
        border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
        lines = [border]
        for index, row in enumerate(rows):
            lines.append(
                "| "
                + " | ".join(
                    value.ljust(width) for value, width in zip(row, widths, strict=True)
                )
                + " |"
            )
            if index == 0:
                lines.append(border)
        lines.append(border)
        return "\n".join(lines)

    rows, matrices = [], []
    for name, value in result.items():
        if (
            isinstance(value, list)
            and value
            and all(isinstance(row, list) for row in value)
            and value[0]
            and all(len(row) == len(value[0]) for row in value)
        ):
            headers = ["Row / Col", *map(str, range(len(value[0])))]
            matrix_rows = [
                [str(index), *map(cell, row)] for index, row in enumerate(value)
            ]
            matrices.append(cell(name) + "\n" + table(headers, matrix_rows))
        else:
            rows.append([cell(name), cell(value)])
    return "\n\n".join([table(["Metric", "Value"], rows), *matrices])


def batch_size(batch) -> int:
    """Tuple batches use targets; task-owned dictionary batches declare their size."""
    from collections.abc import Mapping

    count = batch["batch_size"] if isinstance(batch, Mapping) else len(batch[1])
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Batch size must be a positive integer")
    return count
