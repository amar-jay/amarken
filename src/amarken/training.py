"""Single-device supervised training with immutable resumable checkpoints."""

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from amarken.base import BaseConfig, BaseModel
from amarken.constants import CLINC, DECISION_KINDS, TRAIN_MODES
from amarken.decision import DecisionModel, decision_loss
from amarken.storage import (
    environment,
    fingerprint,
    read_json,
    read_rows,
    verify_prepared,
    write_json,
)


@dataclass(frozen=True)
class TrainConfig:
    mode: str = TRAIN_MODES[0]
    epochs: int | None = None
    effective_batch_size: int = 16
    adapter_learning_rate: float = 2e-4
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    seed: int = 42
    checkpoint_steps: int = 100
    validation_steps: int | None = None
    max_steps: int | None = None
    cpu_smoke: bool = False
    smoke: bool = False

    def __post_init__(self):
        if self.mode not in TRAIN_MODES:
            raise ValueError(f"Training modes: {TRAIN_MODES}")
        if min(self.effective_batch_size, self.checkpoint_steps) < 1:
            raise ValueError("Batch size and checkpoint steps must be positive")
        if self.validation_steps is not None and self.validation_steps < 1:
            raise ValueError("validation_steps must be positive")
        if self.epochs is not None and self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if (
            min(self.adapter_learning_rate, self.head_learning_rate) <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid optimizer settings")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("Invalid warmup fraction")
        if (self.cpu_smoke or self.smoke) and (
            self.max_steps is None or self.max_steps > 10
        ):
            raise ValueError("Smoke runs require max_steps <= 10")


def rng_state():
    return {
        "python": random.getstate(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@torch.no_grad()
def validation_metrics(model, rows):
    model.eval()
    total, correct = 0.0, 0
    for row in rows:
        logits = model(row["state"], row)
        total += float(decision_loss(logits, row))
        if row["kind"] == "noul":
            correct += int((float(logits[0]) >= 0) == (row["target"][0] >= 0.5))
        else:
            correct += int(
                int(logits.argmax())
                == max(range(len(row["target"])), key=row["target"].__getitem__)
            )
    if not rows:
        raise ValueError("Validation split is empty")
    return {"loss": total / len(rows), "accuracy": correct / len(rows)}


def validation_loss(model, rows):
    return validation_metrics(model, rows)["loss"]


def train(
    prepared,
    output,
    config=None,
    base_config=None,
    decision_config=None,
    *,
    resume=None,
    model=None,
    tracking=None,
    backup=None,
):
    config = config or TrainConfig()
    if not config.cpu_smoke and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Repair the local GPU driver before full training; use --cpu-smoke --max-steps N only for small checks."
        )
    manifest = verify_prepared(prepared)
    rows = read_rows(Path(prepared) / "train.jsonl")
    validation = read_rows(Path(prepared) / "validation.jsonl")
    if not rows or not validation:
        raise ValueError("Nonempty training and validation splits are required")
    # CPU smoke bounds BOTH work and validation, clearly separate from real metrics.
    if config.cpu_smoke or config.smoke:
        rows, validation = rows[:32], validation[:2]
    epochs = config.epochs or (
        3 if manifest["data_config"]["dataset_id"] == CLINC else 1
    )
    steps_per_epoch = math.ceil(len(rows) / config.effective_batch_size)
    total_steps = min(
        epochs * steps_per_epoch, config.max_steps or epochs * steps_per_epoch
    )
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = "cpu" if config.cpu_smoke else "cuda"
    dtype = (
        "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    )
    base_config = base_config or BaseConfig(device=device, dtype=dtype)
    if base_config.device != device:
        raise ValueError(f"This training run requires device={device}")
    output = Path(output)
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise ValueError("Resume requires the original run directory")
    prepared_id = fingerprint(Path(prepared) / "manifest.json")
    if model is None:
        model = (
            DecisionModel.load(resume, device=device, dtype=dtype, trainable=True)
            if resume
            else DecisionModel(BaseModel.load(base_config), decision_config)
        )
    if model.base.settings.model_id != base_config.model_id:
        raise ValueError("Requested backbone differs from checkpoint")
    token_meta = manifest.get("tokenizer") or {}
    if token_meta and (
        token_meta.get("model_id"),
        token_meta.get("revision"),
        manifest["prepare_config"]["max_length"],
    ) != (
        model.base.settings.model_id,
        model.base.settings.revision,
        model.base.settings.max_length,
    ):
        raise ValueError(
            "Prepared tokenizer, revision or context limit differs from model"
        )
    model.training_provenance = {
        "prepared_id": prepared_id,
        "files": manifest["files"],
        "smoke_run": config.smoke or config.cpu_smoke,
        "training_mode": config.mode,
        "dataset_id": manifest["data_config"]["dataset_id"],
    }
    model.configure_training(config.mode)
    model.calibration = {
        kind: {"temperature": 1.0, "status": "not_fitted"} for kind in DECISION_KINDS
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.base.parameters() if p.requires_grad],
                "lr": config.adapter_learning_rate,
            },
            {"params": model.head.parameters(), "lr": config.head_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    warmup = max(1, math.ceil(total_steps * config.warmup_fraction))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_epoch, offset, step, best = 0, 0, 0, float("inf")
    history = []
    if resume:
        # Only local checkpoints produced by this pipeline; weights_only excludes arbitrary pickles.
        state = torch.load(
            Path(resume) / "training.pt", map_location="cpu", weights_only=True
        )
        if (
            asdict(TrainConfig(**state["config"])) != asdict(config)
            or state["prepared_id"] != prepared_id
            or state["total_steps"] != total_steps
        ):
            raise ValueError("Resume configuration or prepared data differs")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        # Move optimizer moments after restoring on CPU.
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    values[key] = value.to(device)
        start_epoch, offset, step, best = (
            state["epoch"],
            state["offset"],
            state["step"],
            state["best"],
        )
        history = state["history"]
        restore_rng(state["rng"])
    else:
        write_json(
            output / "run.json",
            {
                "config": asdict(config),
                "base": asdict(base_config),
                "decision": asdict(model.settings),
                "prepared_id": prepared_id,
                "prepared_manifest": manifest,
                "environment": environment(),
                "device": device,
                "dtype": dtype,
                "total_steps": total_steps,
                "hardware": torch.cuda.get_device_name() if device == "cuda" else "cpu",
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            },
        )
        write_json(output / "data-manifest.json", manifest)

    def save_checkpoint(epoch, next_offset, improved=False):
        from amarken.storage import refresh_checksums

        directory = output / f"step-{step:07d}"
        if directory.exists() and resume is not None:
            # A preemption can leave the next checkpoint partially written.
            # Preserve it and retry from the last committed latest.json pointer.
            directory.rename(
                directory.with_name(directory.name + f".incomplete-{time.time_ns()}")
            )
        snapshot_rng = rng_state()
        # Embedded bases are only for injected tiny smoke models, never full LoRA runs.
        model.save(
            directory,
            embed_base=config.cpu_smoke and model.base.model.config.vocab_size < 1000,
        )
        torch.save(
            {
                "config": asdict(config),
                "prepared_id": prepared_id,
                "total_steps": total_steps,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": snapshot_rng,
                "epoch": epoch,
                "offset": next_offset,
                "step": step,
                "best": best,
                "history": history,
            },
            directory / "training.pt",
        )
        write_json(directory / "data-manifest.json", manifest)
        write_json(
            directory / "run.json",
            (
                read_json(output / "run.json")
                if (output / "run.json").exists()
                else {"resumed": True, "config": asdict(config)}
            ),
        )
        refresh_checksums(directory)
        write_json(output / "latest.json", {"checkpoint": directory.name})
        if improved:
            write_json(
                output / "best.json",
                {"checkpoint": directory.name, "validation_loss": best},
            )
        try:
            uploader.checkpoint(output, directory, prepared, step=step)
            tracker.checkpoint(directory, step, improved)
        finally:
            restore_rng(snapshot_rng)

    from amarken.backup import CheckpointBackup
    from amarken.tracking import WandbTracker

    uploader = CheckpointBackup(backup)

    tracking_rng = rng_state()
    tracker = WandbTracker(
        output,
        tracking,
        run_config={
            "training": asdict(config),
            "base": {
                key: value
                for key, value in asdict(base_config).items()
                if key != "cache_dir"
            },
            "decision": asdict(model.settings),
            "dataset": {
                "id": manifest["data_config"]["dataset_id"],
                "revision": manifest["revision"],
                "manifest_sha256": prepared_id,
                "counts": manifest["counts"],
            },
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        },
    )
    restore_rng(tracking_rng)
    success = False
    try:
        for epoch in range(start_epoch, epochs):
            order = list(range(len(rows)))
            random.Random(config.seed + epoch).shuffle(order)
            first = offset if epoch == start_epoch else 0
            for index in range(first, len(order), config.effective_batch_size):
                if step >= total_steps:
                    break
                batch = [
                    rows[i] for i in order[index : index + config.effective_batch_size]
                ]
                model.train()
                # Frozen head baseline has no stochastic encoder behavior.
                if config.mode == "head":
                    model.base.eval()
                optimizer.zero_grad(set_to_none=True)
                start = time.perf_counter()
                losses = [model.backward_question(row, 1 / len(batch)) for row in batch]
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0,
                    error_if_nonfinite=True,
                )
                applied_rates = [group["lr"] for group in optimizer.param_groups]
                optimizer.step()
                scheduler.step()
                step += 1
                entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss": sum(losses) / len(losses),
                    "seconds": time.perf_counter() - start,
                    "grad_norm": float(grad_norm),
                    "adapter_lr": applied_rates[0],
                    "head_lr": applied_rates[1],
                }
                entry["questions_per_second"] = len(batch) / entry["seconds"]
                entry["estimated_remaining_training_seconds"] = (
                    entry["seconds"]
                    / len(batch)
                    * max(0, (total_steps - step) * config.effective_batch_size)
                )
                entry["peak_cuda_bytes"] = (
                    torch.cuda.max_memory_allocated() if device == "cuda" else None
                )
                at_end = index + len(batch) >= len(order) or step == total_steps
                improved = False
                validated = at_end or (
                    config.validation_steps is not None
                    and step % config.validation_steps == 0
                )
                if validated:
                    validation_result = validation_metrics(model, validation)
                    value = validation_result["loss"]
                    entry["validation_loss"] = value
                    entry["validation_accuracy"] = validation_result["accuracy"]
                    if value < best:
                        best, improved = value, True
                history.append(entry)
                tracker.log(entry)
                write_json(output / "history.json", history)
                print(entry, flush=True)
                if validated or step % config.checkpoint_steps == 0:
                    next_offset = index + len(batch)
                    next_epoch = epoch
                    if next_offset >= len(order):
                        next_epoch, next_offset = epoch + 1, 0
                    save_checkpoint(next_epoch, next_offset, improved)
            if step >= total_steps:
                break
        result = read_json(output / "best.json")
        success = True
        return result
    finally:
        tracker.finish(success=success, best=best)
