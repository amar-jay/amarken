"""Shared TRL integration: native state bundles, adapter exports and run metadata."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch


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


def trl_run_metadata(owner, *, trainer_kind, trl_version):
    return {
        "trainer": trainer_kind,
        "trl_version": trl_version,
        "config": asdict(owner.config),
        "run_id": owner.run_id,
        "run_label": owner.run_label,
        "best_metric": owner.engine.state.best_metric,
    }


def mirror_trl_policy_adapter(directory):
    """Keep native PEFT resume compatible when a frozen ref adapter exists."""
    default = directory / "default"
    default.mkdir(exist_ok=True)
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    ):
        if (directory / name).exists():
            shutil.copy2(directory / name, default / name)


def restore_trl_run(owner, path, *, trainer_kind, settings_type, trl_version):
    path = Path(path)
    if owner.engine is not None:
        raise ValueError("Resume requires a fresh TRL adapter")
    if path.is_dir():
        raise ValueError(
            "RL resume requires an amarken .pt checkpoint with complete policy weights"
        )
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if state.get("trainer") != trainer_kind or state.get("trl_version") != trl_version:
        raise ValueError(
            "Resume requires an amarken TRL checkpoint from the installed TRL version"
        )
    previous, current = state["config"], asdict(owner.config)
    # Normalize optional settings added after earlier snapshots.
    previous["rl"] = asdict(settings_type(**previous["rl"]))
    for key in (
        "rl",
        "fine_tuning",
        "model_id",
        "precision",
        "dataset",
        "seed",
        "batch_size",
        "learning_rate",
    ):
        if json.dumps(previous[key], sort_keys=True) != json.dumps(
            current[key], sort_keys=True
        ):
            raise ValueError(f"Resume requires the same {key} settings")
    if any(
        previous["checkpoints"][key] != current["checkpoints"][key]
        for key in ("monitor", "mode")
    ):
        raise ValueError("Resume requires the same checkpoint monitor and mode")
    for name in ("run_id", "run_label"):
        value = state.get(name)
        if value is not None and (
            not value or Path(value).name != value or value in {".", ".."}
        ):
            raise ValueError("Invalid RL checkpoint run identity")
    if state.get("format_version") != 1 or not isinstance(state.get("model"), dict):
        raise ValueError("RL checkpoint is missing complete policy weights")
    # Restore frozen base weights and the trainable policy adapter BEFORE
    # TRL constructs its reference adapter and restores native training state.
    owner.model.load_state_dict(state["model"], strict=True)
    owner.run_id, owner.run_label = state["run_id"], state.get("run_label")
    owner.best_metric = state.get("best_metric")
    owner._resumed = True
    owner.resume_source = path

    return state


def capture_trl_run(owner, *, trainer_kind, trl_version):
    # Inference uses the same full-weight amarken format. TRL's frozen ref
    # adapter stays outside the inference-compatible policy weights.
    weights = {
        key: value
        for key, value in owner.model.state_dict().items()
        if ".ref." not in key
    }
    output = Path(owner.engine.args.output_dir)
    return {
        "format_version": 1,
        **trl_run_metadata(owner, trainer_kind=trainer_kind, trl_version=trl_version),
        "model": weights,
        "epoch": owner.epoch,
        "global_step": owner.global_step,
        "trl_checkpoint": owner.native_checkpoint,
        "trl_files": pack_trl_checkpoints(output),
    }
