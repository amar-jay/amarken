"""Training checkpoint state, resume checks, and rank-specific RNG restoration."""

import random
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from amarken.utils import distributed


def capture_state(trainer, *, training_state=None):
    """Serialize trainer state; FSDP model/optimizer state must be gathered first."""
    if trainer.strategy == "fsdp2" and training_state is None:
        raise ValueError(
            "FSDP2 state capture is collective; pass distributed.full_state(model, optimizer)"
        )
    if training_state is None:
        training_state = (
            distributed.unwrap(trainer.model).state_dict(),
            trainer.optimizer.state_dict(),
        )
    model_state, optimizer_state = training_state
    return {
        "format_version": 1,
        "run_id": trainer.run_id,
        "run_label": trainer.run_label,
        "best_metric": trainer.best_metric,
        "model": model_state,
        "strategy": trainer.strategy,
        "world_size": distributed.world_size(),
        "rank_rng": getattr(trainer, "_rank_rng", None),
        "optimizer": optimizer_state,
        "scheduler": (
            trainer.scheduler.state_dict() if trainer.scheduler is not None else None
        ),
        "scaler": trainer.scaler.state_dict(),
        "epoch": trainer.epoch,
        "global_step": trainer.global_step,
        "config": asdict(trainer.config),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all()
            if trainer.device.type == "cuda" and not distributed.active()
            else None
        ),
    }


def restore_state(trainer, path: str | Path):
    """Validate and restore an epoch checkpoint, including RNG and run identity."""
    restored = {}
    if trainer.strategy == "fsdp2":

        def read_checkpoint():
            restored.update(torch.load(path, map_location="cpu", weights_only=True))
            return {
                key: value
                for key, value in restored.items()
                if key not in {"model", "optimizer"}
            }

        state = distributed.primary_call(read_checkpoint)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("world_size", 1) != distributed.world_size():
        raise ValueError("Resume requires the same world size")
    previous_strategy = state.get(
        "strategy", "ddp" if state.get("world_size", 1) > 1 else "single"
    )
    if previous_strategy != trainer.strategy:
        raise ValueError("Resume requires the same strategy")
    if state["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    if (state["scheduler"] is None) != (trainer.scheduler is None):
        raise ValueError("Resume requires the same scheduler configuration")
    if state["config"]["precision"] != trainer.config.precision:
        raise ValueError("Resume requires the same precision")
    previous = state["config"]["checkpoints"]
    current = trainer.config.checkpoints
    if (previous.get("monitor", "val/loss"), previous.get("mode", "min")) == (
        current.monitor,
        current.mode,
    ):
        trainer.best_metric = state.get("best_metric")
    else:
        trainer.best_metric = None
    run_label = state.get("run_label")
    if run_label is not None and (
        not run_label or Path(run_label).name != run_label or run_label in {".", ".."}
    ):
        raise ValueError("Invalid checkpoint run name")
    trainer.run_label = run_label
    trainer.run_id = state.get("run_id", trainer.run_id)
    if Path(trainer.run_id).name != trainer.run_id or trainer.run_id in {".", "..", ""}:
        raise ValueError("Invalid checkpoint run ID")
    trainer._resumed = True
    if trainer.strategy == "fsdp2":
        distributed.restore_full_state(
            trainer.model,
            trainer.optimizer,
            restored.get("model", {}),
            restored.get("optimizer", {}),
        )
    else:
        distributed.unwrap(trainer.model).load_state_dict(state["model"])
        trainer.optimizer.load_state_dict(state["optimizer"])
    if trainer.scheduler is not None:
        trainer.scheduler.load_state_dict(state["scheduler"])
    trainer.scaler.load_state_dict(state["scaler"])
    trainer.epoch, trainer.global_step = state["epoch"], state["global_step"]
    rng = state
    if state.get("rank_rng") is not None:
        rng = state["rank_rng"][distributed.rank()]
    random.setstate(rng["python_rng"])
    torch.set_rng_state(rng["torch_rng"])
    if rng.get("cuda_local_rng") is not None:
        torch.cuda.set_rng_state(rng["cuda_local_rng"], trainer.device)
    elif rng["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda_rng"])


def gather_rng(trainer):
    """Collect each rank's RNG state before rank zero writes a checkpoint."""
    cuda_rng = None
    if trainer.device.type == "cuda":
        cuda_rng = torch.cuda.get_rng_state(trainer.device)
    rng = {
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": None,
        "cuda_local_rng": cuda_rng,
    }
    if distributed.active():
        states = [None] * distributed.world_size()
        dist.all_gather_object(states, rng)
        trainer._rank_rng = states
