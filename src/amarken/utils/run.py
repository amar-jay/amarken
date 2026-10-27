"""Shared run naming, logger lifetime and cleanup, independent of optimization."""

import sys
import warnings
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from amarken.utils import distributed
from amarken.utils.helpers import prepare_run
from amarken.utils.logger import WandbLogger


def initialize(owner, config, *, logger=None, run_label=None):
    """Initialize only shared run identity and progress bookkeeping."""
    owner.config = config
    owner.logger = logger
    if run_label is not None and (
        not run_label or Path(run_label).name != run_label or run_label in {".", ".."}
    ):
        raise ValueError("Run name must be a single directory name")
    owner.run_label = run_label
    owner._resumed = False
    owner._directory_prepared = False
    owner.run_id = uuid4().hex
    owner.best_metric = None
    owner.epoch = 0
    owner.global_step = 0


def fit(owner, train):
    """Run a training callback with naming, logging and checkpoint cleanup."""

    def prepare():
        prepare_run(owner)
        return owner.run_id, owner.run_label

    owner.run_id, owner.run_label = distributed.primary_call(prepare)
    stack = ExitStack()
    original_logger = owner.logger

    def open_logger():
        if owner.logger is None:
            owner.logger = stack.enter_context(
                WandbLogger(
                    enabled=owner.config.wandb.enabled,
                    mode=owner.config.wandb.mode,
                    name=owner.run_label or owner.run_id,
                    config=asdict(owner.config),
                    checkpoints=owner.config.checkpoints,
                )
            )

    try:
        distributed.primary_call(open_logger)
        history = train()
        distributed.primary_call(lambda: owner.logger.flush_checkpoints())
        distributed.primary_call(stack.close)
        return history
    except BaseException:
        if distributed.primary():
            try:
                if owner.logger is not None:
                    owner.logger.flush_checkpoints()
                stack.__exit__(*sys.exc_info())
            except Exception as error:  # noqa: BLE001 - preserve original failure
                warnings.warn(
                    f"Checkpoint cleanup failed: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        raise
    finally:
        owner.logger = original_logger


def checkpoint_prefix(owner):
    """Use stable paths or one named/UUID directory per run."""
    if owner.config.checkpoints.overwrite:
        return owner.config.run_name
    return f"{owner.config.run_name}/{owner.run_label or owner.run_id}"
