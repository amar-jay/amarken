"""
W&B training logger
Example:
    with WandbLogger(project="experiment", config={"lr": 3e-4}) as logger:
        logger.define_metric("train/*", step_metric="global_step")
        logger.define_metric("val/loss", step_metric="global_step", summary="min")
        for step in range(100):
            logger.log({"train/loss": 1 / (step + 1)}, global_step=step)

This wraps experiment tracking, not every W&B product. `run`, `sdk`, and
`api()` expose the native APIs, including sweeps, plots, registry, and queries.
W&B automatically collects supported system metrics and console output unless
disabled in Settings. Media may require extra dependencies (e.g. Pillow).

DDP: pass the GLOBAL rank; only rank zero initializes/logs. Reduce metrics
across workers yourself. FSDP/ZeRO checkpoints need framework-specific gathering.
Do not combine this logger with another integration that owns the same run
(e.g. Trainer report_to="wandb"). Use one run owner per training process.

Offline: mode="offline", then `wandb sync <run-directory>` after training.
Resume tracking: id="previous-id", resume="must" (online). Restore training
state separately; W&B resume does not reload model/optimizer/RNG state.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from example.constants import CHECKPOINT_DIR, WANDB_CACHE, PROJECT_VERSION, PROJECT_NAME


@dataclass(frozen=True)
class CheckpointConfig:
    """Local checkpoints with optional W&B or S3-compatible replication.

    S3 credentials use boto3's normal environment/profile/IAM credential chain.
    Use a distinct prefix (or filenames) for independent training runs.
    """

    backend: str = "wandb"
    bucket: str | None = None
    prefix: str = "checkpoints"
    endpoint_url: str | None = None
    region_name: str | None = None

    def __post_init__(self):
        if self.backend not in {"local", "wandb", "s3"}:
            raise ValueError("Checkpoint backend must be local, wandb, or s3")
        if self.backend == "s3" and not self.bucket:
            raise ValueError("An S3 bucket is required for checkpoint storage")


class WandbLogger:
    """Own one run; preserve native W&B data and logging semantics.

    Extra keyword arguments go directly to wandb.init (entity, name, tags,
    group, job_type, notes, id, resume, settings, directory via dir, etc.).
    Config supplied by a sweep can be read from logger.config after init.
    SDK errors propagate; training never silently loses logging.
    """

    def __init__(
        self,
        project: str | None = PROJECT_NAME + "v" + PROJECT_VERSION,
        *,
        config: Mapping[str, Any] | None = None,
        enabled: bool = True,
        rank: int | None = None,
        mode: str | None = None,
        checkpoints: CheckpointConfig | Mapping[str, Any] | None = None,
        **init_kwargs: Any,
    ) -> None:
        # RANK is global under torchrun; LOCAL_RANK is insufficient on multi-node.
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else rank
        self.enabled = enabled and self.rank == 0
        self.checkpoints = (CheckpointConfig(**checkpoints)
                            if isinstance(checkpoints, Mapping)
                            else checkpoints or CheckpointConfig())
        self._storage_client: Any = None
        self.run: Any = None
        self._sdk: Any = None
        self._finished = False
        if not self.enabled:
            return

        os.environ.setdefault("WANDB_CACHE_DIR", str(Path(WANDB_CACHE).resolve()))
        import wandb

        self._sdk = wandb
        if wandb.run is not None:
            raise RuntimeError("A W&B run is already active; use one run owner per process.")
        options = dict(init_kwargs)
        if project is not None:
            options["project"] = project
        if config is not None:
            options["config"] = dict(config)
        if mode is not None:
            options["mode"] = mode

        options.setdefault("settings", wandb.Settings(
            disable_code=True,
            x_disable_meta=True,
            x_disable_machine_info=True,
            init_timeout=30,
        ))
        self.run = wandb.init(**options)

    @property
    def sdk(self) -> Any:
        """Native wandb module: Image, Audio, Video, Table, plot, sweep, agent..."""
        if self._sdk is None:
            import wandb
            self._sdk = wandb
        return self._sdk

    @property
    def config(self) -> Any:
        """Authoritative run configuration, including sweep overrides."""
        return self.run.config if self.run is not None else None

    @property
    def active(self) -> bool:
        return self.run is not None and not self._finished

    def _can_write(self) -> bool:
        if not self.enabled:
            return False
        if self._finished:
            raise RuntimeError("This W&B run has already finished.")
        return self.run is not None

    def log(
        self,
        data: Mapping[str, Any],
        *,
        global_step: int | None = None,
        step: int | None = None,
        commit: bool | None = None,
    ) -> None:
        """Log scalars or native W&B objects without converting them.

        global_step is a regular metric for a custom chart axis; define it with
        define_metric. `step` is W&B's internal monotonically increasing step.
        Prefer global_step so validation/media calls do not collide with training
        history. Native step/commit defaults are passed through unchanged.
        """
        if self._can_write():
            payload = dict(data)
            if global_step is not None:
                if "global_step" in payload and payload["global_step"] != global_step:
                    raise ValueError("Conflicting global_step values.")
                payload["global_step"] = global_step
            self.run.log(payload, step=step, commit=commit)

    def define_metric(self, name: str, **kwargs: Any) -> Any:
        if self._can_write():
            return self.run.define_metric(name, **kwargs)

    def update_config(self, values: Mapping[str, Any], *, allow_val_change: bool = False) -> None:
        if self._can_write():
            self.run.config.update(dict(values), allow_val_change=allow_val_change)

    def update_summary(self, values: Mapping[str, Any]) -> None:
        if self._can_write():
            self.run.summary.update(dict(values))

    def watch(self, models: Any, **kwargs: Any) -> None:
        """PyTorch parameter/gradient hooks; choose log_freq to control overhead."""
        if self._can_write():
            self.run.watch(models, **kwargs)

    def unwatch(self, models: Any = None) -> None:
        if self._can_write():
            self.run.unwatch(models)

    def log_media(self, key: str, kind: str, value: Any, *,
                  media_kwargs: Mapping[str, Any] | None = None, **log_kwargs: Any) -> None:
        """kind: Image, Audio, Video, Histogram, Html, Object3D, or Molecule.

        For complex/multiple media objects, build them with logger.sdk and log().
        """
        allowed = {"Image", "Audio", "Video", "Histogram", "Html", "Object3D", "Molecule"}
        if kind not in allowed:
            raise ValueError(f"kind must be one of {sorted(allowed)}")
        if self._can_write():
            self.log({key: getattr(self.sdk, kind)(value, **dict(media_kwargs or {}))}, **log_kwargs)

    def log_table(self, key: str, *, columns: Sequence[str] | None = None,
                  data: Any = None, dataframe: Any = None, **log_kwargs: Any) -> Any:
        if self._can_write():
            if dataframe is not None:
                if columns is not None or data is not None:
                    raise ValueError("Supply dataframe OR columns/data.")
                table = self.sdk.Table(dataframe=dataframe)
            else:
                table = self.sdk.Table(columns=list(columns or []), data=data)
            self.log({key: table}, **log_kwargs)
            return table

    def log_artifact(self, path: str | Path, *, name: str, type: str = "dataset",
                     aliases: Sequence[str] | None = None,
                     metadata: Mapping[str, Any] | None = None,
                     description: str | None = None) -> Any:
        """Version a local file/directory. Return the logged artifact handle."""
        if self._can_write():
            source = Path(path).expanduser().resolve()
            if not source.exists():
                raise FileNotFoundError(source)
            artifact = self.sdk.Artifact(name, type=type, metadata=dict(metadata or {}),
                                         description=description)
            if source.is_dir():
                artifact.add_dir(str(source))
            else:
                artifact.add_file(str(source))
            return self.run.log_artifact(artifact, aliases=list(aliases) if aliases is not None else None)

    def use_artifact(self, name: str, **kwargs: Any) -> Any:
        """Record an input dependency; name may include entity/project/name:alias."""
        if self._can_write():
            return self.run.use_artifact(name, **kwargs)

    def download_artifact(self, name: str, *, root: str | Path | None = None,
                          **kwargs: Any) -> str | None:
        """Rank-zero only. Coordinate other ranks with your framework's barriers."""
        artifact = self.use_artifact(name, **kwargs)
        if artifact is not None:
            return artifact.download(root=str(root) if root is not None else None)
        return None

    def save_checkpoint(self, state: Mapping[str, Any], path: str | Path = "latest.pt", *,
                        name: str | None = None, aliases: Sequence[str] = ("latest",),
                        metadata: Mapping[str, Any] | None = None) -> Any:
        """Save beneath CHECKPOINT_DIR, then replicate to the configured backend.

        `path` is relative to the fixed directory; absolute paths and escapes are
        rejected. Nonzero ranks do nothing. Local/S3 saving works with enabled=False
        (W&B tracking disabled). Returns a Path, S3 URI, or W&B artifact handle.
        Wait on W&B artifact handles before overwriting the same checkpoint.
        Include optimizer/scheduler/RNG state as needed; sharded state must be
        gathered by the caller. Upload failures propagate, retaining the local file.
        """
        if self.rank != 0:
            return None
        if self._finished:
            raise RuntimeError("This logger has already finished.")
        relative = Path(path)
        root = CHECKPOINT_DIR.resolve()
        destination = (root / relative).resolve()
        if relative.is_absolute() or not destination.is_relative_to(root) or destination == root:
            raise ValueError("Checkpoint path must be a file beneath CHECKPOINT_DIR")
        if self.checkpoints.backend == "wandb" and not self._can_write():
            raise RuntimeError("W&B checkpoint storage requires an active run; use local or s3")
        import torch
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Replace atomically so interrupted serialization preserves an older file.
        import tempfile
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(dict(state), temporary_path)
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        if self.checkpoints.backend == "s3":
            if self._storage_client is None:
                import boto3
                self._storage_client = boto3.client(
                    "s3", endpoint_url=self.checkpoints.endpoint_url,
                    region_name=self.checkpoints.region_name,
                )
            prefix = self.checkpoints.prefix.strip("/")
            key = "/".join(filter(None, (prefix, destination.relative_to(root).as_posix())))
            self._storage_client.upload_file(str(destination), self.checkpoints.bucket, key)
            return f"s3://{self.checkpoints.bucket}/{key}"
        if self.checkpoints.backend == "wandb":
            return self.log_artifact(destination, name=name or f"model-{self.run.id}",
                                     type="model", aliases=aliases, metadata=metadata)
        return destination

    def log_code(self, root: str | Path = ".", **kwargs: Any) -> Any:
        """Opt in explicitly; use include_fn/exclude_fn to select source files."""
        if self._can_write():
            return self.run.log_code(root=str(root), **kwargs)

    def alert(self, title: str, text: str, **kwargs: Any) -> None:
        if self._can_write():
            self.run.alert(title=title, text=text, **kwargs)

    def api(self, **kwargs: Any) -> Any:
        """Native Public API; may make network requests even in offline mode."""
        return self.sdk.Api(**kwargs)

    def finish(self, exit_code: int = 0) -> None:
        """Idempotent after a successful finish. Flushes the run through W&B."""
        if self.active:
            self.run.finish(exit_code=exit_code)
            self._finished = True

    def __enter__(self) -> WandbLogger:
        if self._finished:
            raise RuntimeError("Cannot reopen a finished run.")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.finish(exit_code=0 if exc_type is None else 1)
        except Exception as finish_error:
            if exc_type is None:
                raise
            warnings.warn(f"W&B finish failed during training exception: {finish_error}",
                          RuntimeWarning, stacklevel=2)
        return False

