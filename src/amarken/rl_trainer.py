"""TRL GRPO integration; task hooks and amarken checkpoint/logging compatibility.

TRL owns generation, advantages, the objective, optimization, reference adapters
and training-state restoration. Only greedy task evaluation is customized here.
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import trl
from transformers import TrainerCallback
from trl import GRPOConfig
from trl import GRPOTrainer as TRLGRPOTrainer

from amarken.config import RLConfig
from amarken.utils import checkpoint, distributed, run
from amarken.utils import logger as logger_module
from amarken.utils.checkpoint import pack_trl_checkpoints, restore_trl_checkpoints
from amarken.utils.helpers import batch_size, prepare_training_precision


def _metadata(owner):
    return {
        "trainer": "trl-grpo",
        "trl_version": trl.__version__,
        "config": asdict(owner.config),
        "run_id": owner.run_id,
        "run_label": owner.run_label,
        "best_metric": owner.engine.state.best_metric,
    }


class _TaskGRPOTrainer(TRLGRPOTrainer):
    """Use task-owned greedy accuracy for checkpoint selection."""

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        owner = self.task_run
        was_training = self.model.training
        metrics = owner.build_metrics().to(self.accelerator.device)
        samples = 0
        self.model.eval()
        try:
            with torch.inference_mode(), self.accelerator.autocast():
                for batch in owner.validation_loader:
                    owner.validation_step(
                        owner.model,
                        batch,
                        self.accelerator.device,
                        metrics,
                        generation=owner.config.rl.evaluation_generation,
                    )
                    samples += batch_size(batch)
        finally:
            self.model.train(was_training)
        if not samples:
            raise ValueError("RL validation loader must not be empty")
        result = {
            f"{metric_key_prefix}_{key}": value.item()
            for key, value in metrics.compute().items()
        }
        if any(not torch.isfinite(torch.tensor(value)) for value in result.values()):
            raise ValueError("RL validation metrics must be finite scalars")
        monitor = owner.config.checkpoints.monitor
        metric = f"{metric_key_prefix}_{monitor.removeprefix('val/')}"
        if monitor != "train/reward" and metric not in result:
            raise ValueError(
                f"Selected module did not produce checkpoint metric {monitor}"
            )
        result[f"{metric_key_prefix}_selection_score"] = (
            owner.train_metrics.get("reward", 0.0)
            if monitor == "train/reward"
            else result[metric]
        )
        self.log(result)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, result
        )
        return result


class _RunCallback(TrainerCallback):
    def __init__(self, owner):
        self.owner = owner

    def on_step_begin(self, args, state, control, **kwargs):
        self.owner.logger.check_checkpoint_errors()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            aliases = {
                "loss": "loss",
                "reward": "reward",
                "kl": "kl",
                "completions/mean_length": "completion_tokens",
                "completions/clipped_ratio": "truncated_fraction",
                "frac_reward_zero_std": "zero_variance_groups",
                "clip_ratio/region_mean": "clip_fraction",
            }
            self.owner.train_metrics = {
                alias: logs[key] for key, alias in aliases.items() if key in logs
            }

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        owner = self.owner
        owner.global_step, owner.epoch = state.global_step, round(state.epoch)
        row = {f"train/{key}": value for key, value in owner.train_metrics.items()}
        row.update(
            {
                key.replace("eval_", "val/", 1): value
                for key, value in metrics.items()
                if key != "eval_selection_score"
            }
        )
        row["epoch"] = owner.epoch
        owner.history.append(row)
        owner.logger.log(row, global_step=owner.global_step)

    def on_save(self, args, state, control, **kwargs):
        owner = self.owner
        directory = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        # HF resume enumerates adapter subdirectories when TRL's ref adapter
        # exists. Mirror default there too so native resume restores BOTH.
        default = directory / "default"
        default.mkdir(exist_ok=True)
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
        ):
            if (directory / name).exists():
                shutil.copy2(directory / name, default / name)
        (directory / "amarken.json").write_text(json.dumps(_metadata(owner)))
        owner.native_checkpoint = directory.name
        owner._save_checkpoints(owner.history[-1])


class GRPOTrainer:
    """Delegate optimization and precision to TRL; share only run utilities."""

    def __init__(
        self,
        model,
        train_dataset,
        config,
        *,
        validation_loader,
        reward_fn,
        build_metrics,
        validation_step,
    ):
        if config.rl is None:
            raise ValueError("GRPOTrainer requires an rl config")
        if distributed.launched() or distributed.active() or config.strategy != "auto":
            raise ValueError(
                "The TRL integration currently supports one device without torchrun"
            )
        requested = torch.device(config.device)
        if requested.type == "cuda" and (
            requested.index not in (None, 0) or torch.cuda.device_count() != 1
        ):
            raise ValueError(
                "Expose one CUDA device with CUDA_VISIBLE_DEVICES for TRL training"
            )
        if (
            config.checkpoints.monitor != "train/reward"
            and not config.checkpoints.monitor.startswith("val/")
        ):
            raise ValueError(
                "RL checkpoints require train/reward or a module validation metric"
            )
        run.initialize(self, config)
        prepare_training_precision(model, config.precision)
        self.device = requested
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.validation_loader = validation_loader
        self.reward_fn = reward_fn
        self.build_metrics, self.validation_step = build_metrics, validation_step
        self.history, self.train_metrics = [], {}
        self.resume_source = None
        self.native_checkpoint = None
        self.engine = None

    def fit(self):
        return run.fit(self, self._fit)

    @property
    def checkpoint_prefix(self):
        return run.checkpoint_prefix(self)

    def _save_checkpoints(self, metrics):
        return checkpoint.save_checkpoints(self, metrics)

    def resume(self, path):
        path = Path(path)
        if self.engine is not None:
            raise ValueError("Resume requires a fresh GRPOTrainer")
        if path.is_dir():
            raise ValueError(
                "RL resume requires an amarken .pt checkpoint with complete policy weights"
            )
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if (
            state.get("trainer") != "trl-grpo"
            or state.get("trl_version") != trl.__version__
        ):
            raise ValueError(
                "Resume requires an amarken TRL checkpoint from the installed TRL version"
            )
        previous, current = state["config"], asdict(self.config)
        # Older checkpoints predate the optional evaluation budget.
        previous["rl"] = asdict(RLConfig(**previous["rl"]))
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
        self.model.load_state_dict(state["model"], strict=True)
        self.run_id, self.run_label = state["run_id"], state.get("run_label")
        self.best_metric = state.get("best_metric")
        self._resumed = True
        self.resume_source = path

    def state_dict(self, *, training_state=None):
        # Inference uses the same full-weight amarken format. TRL's frozen ref
        # adapter belongs only in the bundled native training state.
        weights = {
            key: value
            for key, value in self.model.state_dict().items()
            if ".ref." not in key
        }
        output = Path(self.engine.args.output_dir)
        return {
            "format_version": 1,
            **_metadata(self),
            "model": weights,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "trl_checkpoint": self.native_checkpoint,
            "trl_files": pack_trl_checkpoints(output),
        }

    def _fit(self):
        run_directory = logger_module.CHECKPOINT_DIR / self.checkpoint_prefix
        run_directory.mkdir(parents=True, exist_ok=True)
        # Stage only in a NEW owned directory. Existing runs are never removed;
        # native TRL state is bundled in best.pt/latest.pt before cleanup.
        with TemporaryDirectory(prefix=".trl-", dir=run_directory) as temporary:
            return self._fit_trl(Path(temporary))

    def _fit_trl(self, output):
        settings = self.config.rl
        native_resume = None
        if self.resume_source is not None:
            native_resume = str(restore_trl_checkpoints(self.resume_source, output))
        args = GRPOConfig(
            output_dir=str(output),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=settings.group_size,
            generation_batch_size=self.config.batch_size * settings.group_size,
            num_generations=settings.group_size,
            num_generations_eval=1,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="constant",
            optim="adamw_torch",
            weight_decay=0.0,
            max_completion_length=settings.max_new_tokens,
            temperature=settings.temperature,
            top_k=0,
            top_p=1.0,
            beta=settings.beta,
            epsilon=settings.clip_epsilon,
            num_iterations=settings.update_epochs,
            loss_type="grpo",
            scale_rewards="group",
            disable_dropout=True,
            gradient_checkpointing=False,
            max_grad_norm=settings.max_grad_norm,
            seed=self.config.seed,
            use_cpu=self.device.type == "cpu",
            bf16=self.config.precision == "bfloat16",
            fp16=self.config.precision == "float16",
            report_to="none",
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="selection_score",
            greater_is_better=self.config.checkpoints.mode == "max",
            dataloader_num_workers=0,
            dataloader_pin_memory=self.device.type == "cuda",
            pad_to_multiple_of=1,
            use_vllm=False,
        )
        self.engine = _TaskGRPOTrainer(
            model=self.model.model,
            reward_funcs=self.reward_fn,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.train_dataset.select(range(1)),
            processing_class=self.model.tokenizer,
            callbacks=[_RunCallback(self)],
        )
        self.engine.task_run = self
        self.logger.define_metric("train/*", step_metric="global_step")
        self.logger.define_metric("val/*", step_metric="global_step")
        self.engine.train(resume_from_checkpoint=native_resume)
        return self.history
