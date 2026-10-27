"""Training checkpoint state, resume checks, and rank-specific RNG restoration."""

import random
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from amarken.utils import distributed


def validate_initial_policy(path, config):
    """Warm-start weights need matching model identity and LoRA scaling/shapes."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise ValueError(
            "Initial policy must be an amarken format_version=1 checkpoint"
        )
    previous = state.get("config", {})
    if previous.get("model_id") != config.model_id:
        raise ValueError("Initial checkpoint model_id does not match the RL config")
    tuning = previous.get("fine_tuning") or {}
    current = asdict(config.fine_tuning)
    for key in ("mode", "r", "alpha", "target_modules"):
        old, new = tuning.get(key), current[key]
        if key == "target_modules":
            old, new = tuple(old or ()), tuple(new)
        if old != new:
            raise ValueError(
                f"Initial checkpoint fine_tuning.{key} does not match the RL config"
            )


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
    previous_tuning = state["config"].get("fine_tuning")
    if state["config"].get("rl") != asdict(trainer.config).get("rl"):
        raise ValueError("Resume requires the same RL settings and trainer type")
    current_tuning = asdict(trainer.config).get("fine_tuning")
    if previous_tuning != current_tuning:
        raise ValueError("Resume requires the same fine_tuning settings")
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
    return state


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


def pack_trl_checkpoints(output):
    """Bundle TRL's retained adapter/optimizer/RNG files into the backend-neutral .pt."""
    return {
        path.relative_to(output).as_posix(): path.read_bytes()
        for directory in output.glob("checkpoint-*")
        if directory.is_dir()
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def restore_trl_checkpoints(path, output):
    """Restore native TRL state from a local or downloaded amarken .pt file."""
    import json

    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    files, current = state.get("trl_files"), state.get("trl_checkpoint")
    if not files or not isinstance(current, str) or Path(current).name != current:
        raise ValueError("RL checkpoint is missing native TRL training state")
    root = output.resolve()
    for name, data in files.items():
        target = (root / name).resolve()
        if (
            Path(name).is_absolute()
            or not target.is_relative_to(root)
            or target == root
        ):
            raise ValueError("Invalid path in native TRL checkpoint")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    # Bundles are portable across machines/run directories. Rebase HF's saved
    # best-model pointer to the retained best checkpoint extracted alongside it.
    for directory in output.glob("checkpoint-*"):
        metadata = directory / "trainer_state.json"
        if metadata.is_file():
            saved = json.loads(metadata.read_text())
            if saved.get("best_model_checkpoint"):
                saved["best_model_checkpoint"] = str(
                    output / Path(saved["best_model_checkpoint"]).name
                )
                metadata.write_text(json.dumps(saved))
    if not (output / current / "trainer_state.json").is_file():
        raise ValueError("Native TRL checkpoint has no trainer state")
    return output / current


def save_checkpoints(owner, metrics, *, capture_training_state=None):
    """Select best/latest and save a snapshot supplied by either trainer.

    Optional collective state capture runs on every rank before rank-zero I/O.
    """
    settings = owner.config.checkpoints
    filenames = []
    if settings.monitor not in metrics:
        raise ValueError(f"Checkpoint monitor {settings.monitor!r} is unavailable")
    value = float(metrics[settings.monitor])
    improved = owner.best_metric is None or (
        value < owner.best_metric
        if settings.mode == "min"
        else value > owner.best_metric
    )
    if improved:
        owner.best_metric = value
        filenames.append(("best.pt", ("best",)))
    if settings.policy == "default":
        filenames.append(("latest.pt", ("latest",)))
    if not filenames:
        return
    training_state = capture_training_state() if capture_training_state else None

    def submit():
        snapshot = owner.state_dict(training_state=training_state)
        for filename, aliases in filenames:
            artifact = owner.logger.save_checkpoint(
                snapshot,
                f"{owner.checkpoint_prefix}/{filename}",
                name=f"model-{owner.run_id}",
                aliases=aliases,
                # Update best/latest within the selected run directory.
                overwrite=True,
            )
            if (
                not settings.async_save
                and settings.backend == "wandb"
                and artifact is not None
            ):
                artifact.wait()

    distributed.primary_call(submit)
