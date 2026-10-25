"""Optional W&B metrics and explicitly requested weights-only artifacts."""

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from amarken.constants import WANDB_CACHE, WANDB_MODES, WANDB_PROJECT
from amarken.storage import fingerprint, read_json, write_json


@dataclass(frozen=True)
class WandbConfig:
    mode: str = WANDB_MODES[0]
    project: str = WANDB_PROJECT
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    log_checkpoints: bool = False

    def __post_init__(self):
        if self.mode not in WANDB_MODES:
            raise ValueError(f"W&B modes: {WANDB_MODES}")
        if not self.project.strip():
            raise ValueError("W&B project must be nonempty")


class WandbTracker:
    def __init__(self, output, config=None, *, run_config=None):
        self.settings = config or WandbConfig()
        self.run = None
        self.output = Path(output)
        if self.settings.mode == "disabled":
            return
        os.environ.setdefault("WANDB_CACHE_DIR", str(Path(WANDB_CACHE).resolve()))
        import wandb

        identity_path = self.output / "wandb-run.json"
        identity = {"project": self.settings.project, "entity": self.settings.entity}
        previous = read_json(identity_path) if identity_path.exists() else None
        if previous and previous["identity"] != identity:
            raise ValueError(
                "W&B project/entity differs from this run's recorded identity"
            )
        run_id = previous["id"] if previous else uuid4().hex[:8]
        directory = self.output
        directory.mkdir(exist_ok=True)
        self.run = wandb.init(
            project=self.settings.project,
            entity=self.settings.entity,
            name=self.settings.name or self.output.name,
            group=self.settings.group,
            id=run_id,
            resume="allow" if self.settings.mode == "online" else None,
            mode=self.settings.mode,
            dir=str(directory.resolve()),
            config=run_config,
            save_code=False,
            settings=wandb.Settings(
                disable_code=True,
                disable_git=True,
                console="off",
                x_disable_meta=True,
                x_disable_machine_info=True,
                init_timeout=30,
            ),
        )
        self.run.define_metric("train/step")
        for pattern in ("train/*", "validation/*", "optimizer/*", "performance/*"):
            self.run.define_metric(pattern, step_metric="train/step")
        self.run.define_metric(
            "validation/loss", step_metric="train/step", summary="min"
        )
        write_json(
            identity_path,
            {
                "identity": identity,
                "id": run_id,
                "mode": self.settings.mode,
                "url": self.run.url if self.settings.mode == "online" else None,
            },
        )

    def log(self, entry):
        if self.run is None:
            return
        data = {
            "train/step": entry["step"],
            "train/epoch": entry["epoch"],
            "train/loss": entry["loss"],
            "performance/step_seconds": entry["seconds"],
        }
        for key, destination in (
            ("validation_loss", "validation/loss"),
            ("validation_accuracy", "validation/accuracy"),
            ("grad_norm", "train/grad_norm"),
            ("adapter_lr", "optimizer/adapter_lr"),
            ("head_lr", "optimizer/head_lr"),
            ("questions_per_second", "performance/questions_per_second"),
            ("peak_cuda_bytes", "performance/peak_cuda_bytes"),
            (
                "estimated_remaining_training_seconds",
                "performance/estimated_remaining_training_seconds",
            ),
        ):
            if entry.get(key) is not None:
                data[destination] = entry[key]
        # Use a custom x-axis instead of SDK step so checkpoint rollbacks do not
        # cause W&B to discard replayed training steps.
        self.run.log(data)

    def checkpoint(self, directory, step, improved):
        if self.run is None or not self.settings.log_checkpoints:
            return
        import wandb

        directory = Path(directory)
        artifact = wandb.Artifact(
            f"amarken-{self.run.id}", type="model", metadata={"step": step}
        )
        files = [
            directory / name for name in ("head.pt", "model.json", "calibration.json")
        ]
        for name in ("adapter", "tokenizer", "base"):
            if (directory / name).is_dir():
                files.extend(p for p in (directory / name).rglob("*") if p.is_file())
        checksums = {}
        for path in files:
            name = str(path.relative_to(directory))
            checksums[name] = fingerprint(path)
            artifact.add_file(str(path), name=name)
        manifest_path = (
            self.output / "wandb-artifact-manifests" / directory.name / "checksums.json"
        )
        write_json(manifest_path, checksums)
        artifact.add_file(str(manifest_path), name="checksums.json")
        # Do not upload optimizer state, training rows, run.json (source snapshots),
        # environment dumps or data manifests. Downloaded weights remain loadable.
        self.run.log_artifact(
            artifact, aliases=["latest", "best"] if improved else ["latest"]
        )

    def finish(self, *, success, best=None):
        if self.run is None:
            return
        try:
            if best is not None and math.isfinite(best):
                self.run.summary["best_validation_loss"] = best
            self.run.finish(exit_code=0 if success else 1)
        except Exception:
            if success:
                raise
            warnings.warn(
                "W&B finalization failed while training was failing; local checkpoints are retained.",
                stacklevel=2,
            )
