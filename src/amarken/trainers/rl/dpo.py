"""TRL DPO adapter for preference pairs; no rollout or reward callbacks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import trl
from transformers import TrainerCallback
from trl import DPOConfig as TRLDPOConfig
from trl import DPOTrainer as TRLDPOTrainer

from amarken.config import DPOConfig
from amarken.utils import checkpoint, distributed, run
from amarken.utils import logger as logger_module
from amarken.utils import trl as trl_utils
from amarken.utils.helpers import prepare_training_precision


def _metrics(values, prefix):
    return {
        f"{prefix}/{name.removeprefix('eval_')}": float(value)
        for name, value in values.items()
        if name
        not in {
            "epoch",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
        }
        and isinstance(value, (int, float))
    }


class _RunCallback(TrainerCallback):
    def __init__(self, owner):
        self.owner = owner

    def on_step_begin(self, args, state, control, **kwargs):
        self.owner.logger.check_checkpoint_errors()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.owner.train_metrics = _metrics(logs, "train")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        owner = self.owner
        owner.global_step, owner.epoch = state.global_step, round(state.epoch)
        row = {**owner.train_metrics, **_metrics(metrics, "val"), "epoch": owner.epoch}
        owner.history.append(row)
        owner.logger.log(row, global_step=owner.global_step)

    def on_save(self, args, state, control, **kwargs):
        owner = self.owner
        directory = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        trl_utils.mirror_trl_policy_adapter(directory)
        metadata = trl_utils.trl_run_metadata(
            owner, trainer_kind="trl-dpo", trl_version=trl.__version__
        )
        (directory / "amarken.json").write_text(json.dumps(metadata))
        owner.native_checkpoint = directory.name
        owner._save_checkpoints(owner.history[-1])


class DPOTrainer:
    """Optimize chosen/rejected responses against the fixed initial LoRA policy."""

    config_type = DPOConfig

    @classmethod
    def from_config(cls, model, config):
        from amarken.dataset import build_preference_dataset

        if not isinstance(config.rl, cls.config_type):
            raise TypeError("Select the matching RL adapter and its configuration")
        train, validation = build_preference_dataset(
            config.dataset,
            tokenize=model.tokenize,
            max_length=config.rl.max_length,
            seed=config.seed,
        )
        return cls(model, train, config, validation_dataset=validation)

    def __init__(self, model, train_dataset, config, *, validation_dataset):
        if not isinstance(config.rl, DPOConfig):
            raise TypeError("DPOTrainer requires DPOConfig settings")
        if distributed.launched() or distributed.active() or config.strategy != "auto":
            raise ValueError(
                "DPO currently supports one device without torchrun, DDP or FSDP"
            )
        device = torch.device(config.device)
        if device.type == "cuda" and (
            device.index not in (None, 0) or torch.cuda.device_count() != 1
        ):
            raise ValueError("Expose one CUDA device with CUDA_VISIBLE_DEVICES for DPO")
        if not config.checkpoints.monitor.startswith("val/"):
            raise ValueError("DPO checkpoints require a val/ metric such as val/loss")
        if not hasattr(model.model, "peft_config"):
            raise TypeError("DPO currently requires a LoRA policy")
        if config.rl.max_length > model.model.config.max_position_embeddings:
            raise ValueError("DPO max_length exceeds the model context window")
        run.initialize(self, config)
        prepare_training_precision(model, config.precision)
        self.model = model.to(device)
        self.device = device
        self.train_dataset, self.validation_dataset = train_dataset, validation_dataset
        self.engine = None
        self.resume_source = None
        self.reference_adapter = None
        self.native_checkpoint = None
        self.history, self.train_metrics = [], {}

    def fit(self):
        return run.fit(self, self._fit)

    @property
    def checkpoint_prefix(self):
        return run.checkpoint_prefix(self)

    def _save_checkpoints(self, metrics):
        return checkpoint.save_checkpoints(self, metrics)

    def state_dict(self, *, training_state=None):
        state = trl_utils.capture_trl_run(
            self, trainer_kind="trl-dpo", trl_version=trl.__version__
        )
        state["reference_adapter"] = {
            key: value
            for key, value in self.model.state_dict().items()
            if ".ref." in key
        }
        return state

    def resume(self, path):
        state = trl_utils.restore_trl_run(
            self,
            path,
            trainer_kind="trl-dpo",
            settings_type=DPOConfig,
            trl_version=trl.__version__,
        )
        self._read_reference(state)

    def _read_reference(self, state):
        reference = state.get("reference_adapter")
        if (
            state.get("trainer") != "trl-dpo"
            or not isinstance(reference, dict)
            or not reference
        ):
            raise ValueError("DPO checkpoint is missing the frozen reference adapter")
        self.reference_adapter = reference

    def _restore_reference(self):
        if self.reference_adapter is None:
            return
        destination = {
            key: value
            for key, value in self.model.state_dict().items()
            if ".ref." in key
        }
        if destination.keys() != self.reference_adapter.keys():
            raise ValueError("DPO reference adapter does not match the selected model")
        with torch.no_grad():
            for key, value in self.reference_adapter.items():
                if destination[key].shape != value.shape:
                    raise ValueError("DPO reference adapter shapes do not match")
                destination[key].copy_(value)

    def _create_engine(self, output, *, training):
        settings = self.config.rl
        args = TRLDPOConfig(
            output_dir=str(output),
            beta=settings.beta,
            loss_type="sigmoid",
            max_length=settings.max_length,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            lr_scheduler_type="constant",
            optim="adamw_torch",
            weight_decay=0.0,
            max_grad_norm=settings.max_grad_norm,
            gradient_checkpointing=settings.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            disable_dropout=True,
            seed=self.config.seed,
            use_cpu=self.device.type == "cpu",
            bf16=self.config.precision == "bfloat16",
            fp16=self.config.precision == "float16",
            report_to="none",
            logging_strategy="epoch",
            eval_strategy="epoch" if training else "no",
            save_strategy="epoch" if training else "no",
            save_total_limit=2,
            load_best_model_at_end=training,
            metric_for_best_model=self.config.checkpoints.monitor.removeprefix("val/"),
            greater_is_better=self.config.checkpoints.mode == "max",
            dataloader_num_workers=0,
            dataloader_pin_memory=self.device.type == "cuda",
            pad_to_multiple_of=1,
        )
        self.engine = TRLDPOTrainer(
            model=self.model.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            processing_class=self.model.tokenizer,
            callbacks=[_RunCallback(self)] if training else [],
        )
        self._restore_reference()
        return self.engine

    def _fit(self):
        directory = logger_module.CHECKPOINT_DIR / self.checkpoint_prefix
        directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".trl-", dir=directory) as output:
            native_resume = None
            if self.resume_source is not None:
                native_resume = str(
                    trl_utils.restore_trl_checkpoints(self.resume_source, Path(output))
                )
            engine = self._create_engine(output, training=True)
            self.logger.define_metric("train/*", step_metric="global_step")
            self.logger.define_metric("val/*", step_metric="global_step")
            engine.train(resume_from_checkpoint=native_resume)
        return self.history

    @classmethod
    def evaluate_checkpoint(cls, model, config, path, *, split):
        from amarken.dataset import build_preference_dataset

        records = build_preference_dataset(
            config.dataset,
            tokenize=model.tokenize,
            max_length=config.rl.max_length,
            seed=config.seed,
            split=split,
        )
        owner = cls(model, records, config, validation_dataset=records)
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        owner._read_reference(state)
        with TemporaryDirectory(prefix="amarken-dpo-eval-") as output:
            values = owner._create_engine(output, training=False).evaluate()
        result = {"samples": len(records), **_metrics(values, split)}
        if any(
            not torch.isfinite(torch.tensor(value)).item() for value in result.values()
        ):
            raise ValueError("DPO evaluation metrics must be finite")
        return result
