"""TRL GRPO integration; task hooks and amarken checkpoint/logging compatibility.

TRL owns generation, advantages, the objective, optimization, reference adapters
and training-state restoration. Only greedy task evaluation is customized here.
"""

import torch
from trl import GRPOConfig as TRLGRPOConfig
from trl import GRPOTrainer as TRLGRPOTrainer

from amarken.config import GRPOConfig
from amarken.trainers.rl.base import BaseRLTrainer, RLCallback
from amarken.utils.helpers import batch_size


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
                        **owner.evaluation_options(owner.config),
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


class GRPOTrainer(BaseRLTrainer):
    """GRPO-specific inputs, metrics and native engine configuration."""

    config_type = GRPOConfig
    trainer_kind = "trl-grpo"

    @classmethod
    def from_config(cls, model, config):
        from amarken.dataset import build_rl_dataset
        from amarken.model import build_metrics, get_rl_hooks, validation_step

        if not isinstance(config.rl, cls.config_type):
            raise TypeError("Select the matching RL adapter and its configuration")
        hooks = get_rl_hooks()
        records, evaluation = build_rl_dataset(
            config.dataset,
            seed=config.seed,
            batch_size=config.batch_size,
            tokenize=getattr(model, "tokenize", None),
            prepare_prompts=hooks["prepare_prompts"],
            max_new_tokens=config.rl.max_new_tokens,
        )
        return cls(
            model,
            records,
            config,
            validation_loader=evaluation,
            reward_fn=hooks["reward_fn"],
            build_metrics=build_metrics,
            validation_step=validation_step,
        )

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
        super().__init__(model, config)
        self.train_dataset = train_dataset
        self.validation_loader = validation_loader
        self.reward_fn = reward_fn
        self.build_metrics, self.validation_step = build_metrics, validation_step

    def validate_config(self, model, config):
        if config.fine_tuning is None or config.fine_tuning.mode != "lora":
            raise ValueError("GRPO currently requires fine_tuning.mode=lora")
        if (
            config.checkpoints.monitor != "train/reward"
            and not config.checkpoints.monitor.startswith("val/")
        ):
            raise ValueError(
                "RL checkpoints require train/reward or a module validation metric"
            )

    @classmethod
    def evaluation_options(cls, config):
        return {"generation": config.rl.evaluation_generation}

    def training_metrics(self, logs):
        aliases = {
            "loss": "loss",
            "reward": "reward",
            "kl": "kl",
            "completions/mean_length": "completion_tokens",
            "completions/clipped_ratio": "truncated_fraction",
            "frac_reward_zero_std": "zero_variance_groups",
            "clip_ratio/region_mean": "clip_fraction",
        }
        return {alias: logs[key] for key, alias in aliases.items() if key in logs}

    def evaluation_metrics(self, metrics):
        result = super().evaluation_metrics(metrics)
        result.pop("selection_score", None)
        return result

    def build_engine(self, output, *, training=True):
        settings = self.config.rl
        args = TRLGRPOConfig(
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
            eval_strategy="epoch" if training else "no",
            save_strategy="epoch" if training else "no",
            save_total_limit=2,
            load_best_model_at_end=training,
            metric_for_best_model="selection_score",
            greater_is_better=self.config.checkpoints.mode == "max",
            dataloader_num_workers=0,
            dataloader_pin_memory=self.device.type == "cuda",
            pad_to_multiple_of=1,
            use_vllm=False,
        )
        engine = _TaskGRPOTrainer(
            model=self.model.model,
            reward_funcs=self.reward_fn,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.train_dataset.select(range(1)),
            processing_class=self.model.tokenizer,
            callbacks=[RLCallback(self)] if training else [],
        )
        engine.task_run = self
        return engine
