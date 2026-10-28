"""TRL DPO adapter for preference pairs; no rollout or reward callbacks."""

from tempfile import TemporaryDirectory

import torch
from trl import DPOConfig as TRLDPOConfig
from trl import DPOTrainer as TRLDPOTrainer

from amarken.config import DPOConfig
from amarken.trainers.rl.base import BaseRLTrainer, RLCallback


class DPOTrainer(BaseRLTrainer):
    """Optimize chosen/rejected responses against the fixed initial LoRA policy."""

    config_type = DPOConfig
    trainer_kind = "trl-dpo"

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
        super().__init__(model, config)
        self.train_dataset, self.validation_dataset = train_dataset, validation_dataset
        self.reference_adapter = None

    def validate_config(self, model, config):
        if not config.checkpoints.monitor.startswith("val/"):
            raise ValueError("DPO checkpoints require a val/ metric such as val/loss")
        if not hasattr(model.model, "peft_config"):
            raise TypeError("DPO currently requires a LoRA policy")
        if config.rl.max_length > model.model.config.max_position_embeddings:
            raise ValueError("DPO max_length exceeds the model context window")

    def extra_state_dict(self):
        return {
            "reference_adapter": {
                key: value
                for key, value in self.model.state_dict().items()
                if ".ref." in key
            }
        }

    def restore_extra_state(self, state):
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

    def build_engine(self, output, *, training=True):
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
        engine = TRLDPOTrainer(
            model=self.model.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            processing_class=self.model.tokenizer,
            callbacks=[RLCallback(self)] if training else [],
        )
        self._restore_reference()
        return engine

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
            owner.engine = owner.build_engine(output, training=False)
            values = owner.engine.evaluate()
        result = {
            "samples": len(records),
            **{
                f"{split}/{key}": value
                for key, value in owner.evaluation_metrics(values).items()
            },
        }
        if any(
            not torch.isfinite(torch.tensor(value)).item() for value in result.values()
        ):
            raise ValueError("DPO evaluation metrics must be finite")
        return result
