"""Shared lifecycle for RL adapters using native TRL training engines."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import trl
from transformers import TrainerCallback

from amarken.utils import checkpoint, distributed, run
from amarken.utils import logger as logger_module
from amarken.utils import trl as trl_utils
from amarken.utils.helpers import prepare_training_precision


class RLCallback(TrainerCallback):
    """Record adapter metrics and export native TRL saves through the shared logger."""

    def __init__(self, owner):
        self.owner = owner

    def on_step_begin(self, args, state, control, **kwargs):
        self.owner.logger.check_checkpoint_errors()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.owner.train_metrics = self.owner.training_metrics(logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        owner = self.owner
        owner.global_step, owner.epoch = state.global_step, round(state.epoch)
        row = {f"train/{key}": value for key, value in owner.train_metrics.items()}
        row.update(
            {
                f"val/{key}": value
                for key, value in owner.evaluation_metrics(metrics).items()
            }
        )
        row["epoch"] = owner.epoch
        owner.history.append(row)
        owner.logger.log(row, global_step=owner.global_step)

    def on_save(self, args, state, control, **kwargs):
        directory = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self.owner.save_native_checkpoint(directory)


class BaseRLTrainer(ABC):
    """Share run/state mechanics; adapters own inputs, objectives and evaluation.

    Subclasses declare config_type and trainer_kind and implement from_config()
    and build_engine(). TRL owns optimization and scaling; this class owns neither.
    """

    config_type: type
    trainer_kind: str

    def __init__(self, model, config):
        if not isinstance(config.rl, self.config_type):
            raise TypeError("Select the matching RL adapter and its configuration")
        if distributed.launched() or distributed.active() or config.strategy != "auto":
            raise ValueError(
                "The TRL integration currently supports one device without torchrun, DDP or FSDP"
            )
        device = torch.device(config.device)
        if device.type == "cuda" and (
            device.index not in (None, 0) or torch.cuda.device_count() != 1
        ):
            raise ValueError(
                "Expose one CUDA device with CUDA_VISIBLE_DEVICES for TRL training"
            )
        self.validate_config(model, config)
        run.initialize(self, config)
        prepare_training_precision(model, config.precision)
        self.device = device
        self.model = model.to(device)
        self.engine = None
        self.resume_source = None
        self.native_checkpoint = None
        self.history, self.train_metrics = [], {}

    @classmethod
    @abstractmethod
    def from_config(cls, model, config):
        """Prepare algorithm-specific inputs and return a configured adapter."""

    @abstractmethod
    def build_engine(self, output, *, training=True):
        """Return a native engine; attach RLCallback(self) for training saves/logs."""

    def validate_config(self, model, config):
        """Optionally validate algorithm-specific requirements before placement."""

    @classmethod
    def evaluation_options(cls, config):
        """Optional keyword arguments for the selected module's validation step."""
        return {}

    @classmethod
    def evaluate_checkpoint(cls, model, config, path, *, split):
        """Use ordinary task evaluation unless the adapter needs another path."""
        from amarken.validation import evaluate_model

        if not isinstance(config.rl, cls.config_type):
            raise TypeError("Select the matching RL adapter and its configuration")
        return evaluate_model(
            model,
            config,
            split=split,
            evaluation_options=cls.evaluation_options(config),
        )

    def fit(self):
        return run.fit(self, self._fit)

    @property
    def checkpoint_prefix(self):
        return run.checkpoint_prefix(self)

    def metadata(self):
        return trl_utils.trl_run_metadata(
            self, trainer_kind=self.trainer_kind, trl_version=trl.__version__
        )

    def _save_checkpoints(self, metrics):
        return checkpoint.save_checkpoints(self, metrics)

    def state_dict(self, *, training_state=None):
        state = trl_utils.capture_trl_run(
            self, trainer_kind=self.trainer_kind, trl_version=trl.__version__
        )
        state.update(self.extra_state_dict())
        return state

    def extra_state_dict(self):
        """Return additional algorithm state for the public checkpoint."""
        return {}

    def resume(self, path):
        state = trl_utils.restore_trl_run(
            self,
            path,
            trainer_kind=self.trainer_kind,
            settings_type=self.config_type,
            trl_version=trl.__version__,
        )
        self.restore_extra_state(state)

    def restore_extra_state(self, state):
        """Optionally restore algorithm state after complete policy restoration."""

    def training_metrics(self, logs):
        """Return unprefixed training metric names and scalar values."""
        return {
            key: value
            for key, value in logs.items()
            if key != "epoch" and isinstance(value, (int, float))
        }

    def evaluation_metrics(self, metrics):
        """Return unprefixed evaluation metrics, excluding runtime bookkeeping."""
        excluded = {
            "epoch",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
        }
        return {
            key.removeprefix("eval_"): value
            for key, value in metrics.items()
            if key not in excluded and isinstance(value, (int, float))
        }

    def save_native_checkpoint(self, directory):
        trl_utils.mirror_trl_policy_adapter(directory)
        (directory / "amarken.json").write_text(json.dumps(self.metadata()))
        self.native_checkpoint = directory.name
        self._save_checkpoints(self.history[-1])

    def train(self, engine, *, resume_from_checkpoint=None):
        """Override only when invoking the native engine needs extra behavior."""
        return engine.train(resume_from_checkpoint=resume_from_checkpoint)

    def _fit(self):
        directory = logger_module.CHECKPOINT_DIR / self.checkpoint_prefix
        directory.mkdir(parents=True, exist_ok=True)
        # Only this newly owned staging directory is removed after export.
        with TemporaryDirectory(prefix=".trl-", dir=directory) as output:
            native_resume = None
            if self.resume_source is not None:
                native_resume = str(
                    trl_utils.restore_trl_checkpoints(self.resume_source, Path(output))
                )
            self.engine = self.build_engine(Path(output), training=True)
            self.logger.define_metric("train/*", step_metric="global_step")
            self.logger.define_metric("val/*", step_metric="global_step")
            self.train(self.engine, resume_from_checkpoint=native_resume)
        return self.history
