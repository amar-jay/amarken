"""PyTorch Lightning training entry point for all Amarken causal LMs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
import torch
from torch import Tensor

from src.models.common import AmarkenCausalLM
from src.models.factory import create_config, create_model
from src.tokenization.tokenizer import load_tokenizer
from src.data.dataset import AmarkenDataset, PackedConversationDataset
from src.training.optimizer import OptimizerConfig, create_optimizer


@dataclass(frozen=True)
class TrainerConfig:
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    seed: int = 2026
    log_every_steps: int = 10
    audit_every_steps: int = 1
    checkpoint_every_steps: int = 1_000
    validation_every_steps: int = 1_000
    num_workers: int = 0
    output_dir: Path = Path("runs/default")
    lr_schedule: Literal["constant", "cosine"] = "constant"
    warmup_steps: int = 0
    total_steps: int | None = None
    min_lr_ratio: float = 0.1
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled"
    wandb_log_model: bool = False

    def __post_init__(self) -> None:
        if min(self.batch_size, self.gradient_accumulation_steps, self.log_every_steps,
               self.audit_every_steps,
               self.checkpoint_every_steps, self.validation_every_steps) < 1:
            raise ValueError("batch sizes and step intervals must be positive")
        if self.num_workers < 0 or self.max_grad_norm <= 0:
            raise ValueError("num_workers or max_grad_norm is invalid")
        if self.warmup_steps < 0 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("invalid warmup_steps or min_lr_ratio")
        if self.lr_schedule == "cosine" and (
            self.total_steps is None or self.total_steps <= self.warmup_steps
        ):
            raise ValueError("cosine schedule requires total_steps > warmup_steps")
        if self.wandb_mode != "disabled" and not self.wandb_project:
            raise ValueError("wandb_project is required when W&B is enabled")
        if self.wandb_mode == "offline" and self.wandb_log_model:
            raise ValueError("wandb_log_model requires online W&B mode")


class AmarkenDataModule(L.LightningDataModule):
    """Distributed-safe streaming data module for the synthetic chat shards."""

    def __init__(self, *, tokenizer_path: str | Path, train_data: str | Path,
                 validation_data: str | Path, sequence_length: int, batch_size: int,
                 seed: int = 2026, train_split: str = "train",
                 validation_split: str = "validation", train_token_budget: int | None = None,
                 validation_token_budget: int | None = None, num_workers: int = 0,
                 shuffle_buffer_size: int = 10_000) -> None:
        super().__init__()
        self.tokenizer_path, self.train_data = Path(tokenizer_path), Path(train_data)
        self.validation_data = Path(validation_data)
        self.sequence_length, self.batch_size, self.seed = sequence_length, batch_size, seed
        self.train_split, self.validation_split = train_split, validation_split
        self.train_token_budget, self.validation_token_budget = train_token_budget, validation_token_budget
        self.num_workers, self.shuffle_buffer_size = num_workers, shuffle_buffer_size
        self.train_dataset: PackedConversationDataset | None = None
        self.validation_dataset: PackedConversationDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        tokenizer = load_tokenizer(self.tokenizer_path)
        if stage in (None, "fit"):
            self.train_dataset = PackedConversationDataset(
                AmarkenDataset(self.train_data, self.train_split, shuffle=True, seed=self.seed,
                               shuffle_buffer_size=self.shuffle_buffer_size),  # type: ignore[arg-type]
                tokenizer, self.sequence_length, token_budget=self.train_token_budget)
            self.validation_dataset = PackedConversationDataset(
                AmarkenDataset(self.validation_data, self.validation_split),  # type: ignore[arg-type]
                tokenizer, self.sequence_length, token_budget=self.validation_token_budget)

    def _loader(self, dataset: PackedConversationDataset) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=self.num_workers > 0)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("setup('fit') must run before train_dataloader")
        return self._loader(self.train_dataset)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self.validation_dataset is None:
            raise RuntimeError("setup('fit') must run before val_dataloader")
        return self._loader(self.validation_dataset)


class AmarkenLightningModule(L.LightningModule):
    """Thin Lightning adapter that keeps architecture code framework-agnostic."""

    def __init__(self, model: AmarkenCausalLM,
                 optimizer_config: OptimizerConfig = OptimizerConfig(), *,
                 lr_schedule: Literal["constant", "cosine"] = "constant",
                 warmup_steps: int = 0, total_steps: int | None = None,
                 min_lr_ratio: float = 0.1, gradient_checkpointing: bool = True,
                 audit_every_steps: int = 1) -> None:
        super().__init__()
        if lr_schedule == "cosine" and (total_steps is None or total_steps <= warmup_steps):
            raise ValueError("cosine schedule requires total_steps > warmup_steps")
        self.model, self.optimizer_config = model, optimizer_config
        self.lr_schedule, self.warmup_steps = lr_schedule, warmup_steps
        self.total_steps, self.min_lr_ratio = total_steps, min_lr_ratio
        if audit_every_steps < 1:
            raise ValueError("audit_every_steps must be positive")
        self.audit_every_steps = audit_every_steps
        self._pending_micro_batches: list[dict[str, float | int]] = []
        self._pending_audit: dict[str, Any] | None = None
        self.model.set_gradient_checkpointing(gradient_checkpointing)
        self.save_hyperparameters({
            "model_type": model.config.model_type, "model_config": asdict(model.config),
            "optimizer": asdict(optimizer_config), "lr_schedule": lr_schedule,
            "warmup_steps": warmup_steps, "total_steps": total_steps,
            "min_lr_ratio": min_lr_ratio, "gradient_checkpointing": gradient_checkpointing,
            "audit_every_steps": audit_every_steps})

    def forward(self, input_ids: Tensor, **kwargs: Tensor):
        return self.model(input_ids, **kwargs)

    @staticmethod
    def _model_inputs(batch: dict[str, Tensor]) -> dict[str, Tensor | None]:
        single_segment = all(row[row >= 0].unique().numel() <= 1 for row in batch["segment_ids"])
        if bool(batch["attention_mask"].all()) and single_segment:
            return {"attention_mask": None, "segment_ids": None}
        return {"attention_mask": batch["attention_mask"], "segment_ids": batch["segment_ids"]}

    def _validate_training_batch(self, batch: dict[str, Tensor]) -> None:
        """Reject malformed packed data before it can produce a model update."""
        required = {"input_ids", "labels", "attention_mask", "segment_ids"}
        if set(batch) != required:
            raise ValueError(f"training batch keys must be {sorted(required)}")
        input_ids, labels = batch["input_ids"], batch["labels"]
        valid, segments = batch["attention_mask"].bool(), batch["segment_ids"]
        if any(tensor.shape != input_ids.shape for tensor in (labels, valid, segments)):
            raise ValueError("all training batch tensors must share [batch, sequence] shape")
        if input_ids.ndim != 2 or input_ids.numel() == 0:
            raise ValueError("training input_ids must be a nonempty rank-2 tensor")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= self.model.config.vocab_size:
            raise ValueError("input token ID is outside the model vocabulary")
        if not bool(labels[~valid].eq(-100).all()) or not bool(segments[~valid].eq(-1).all()):
            raise ValueError("padding must use label -100 and segment ID -1")
        supervised = labels.ne(-100)
        if not bool(torch.equal(labels[supervised], input_ids[supervised])):
            raise ValueError("supervised causal-LM labels must equal their input tokens")
        if not bool((segments[valid] >= 0).all()):
            raise ValueError("valid tokens require nonnegative segment IDs")

    @staticmethod
    def _tensor_summary(parameters: list[Tensor], *, gradients: bool) -> dict[str, float | int | bool]:
        """Return auditable health statistics without serializing model tensors."""
        squared_norm = 0.0
        max_abs = 0.0
        tensors = nonfinite_tensors = elements = 0
        for parameter in parameters:
            tensor = parameter.grad if gradients else parameter
            if tensor is None:
                continue
            detached = tensor.detach()
            tensors += 1
            elements += detached.numel()
            finite = torch.isfinite(detached)
            if not bool(finite.all()):
                nonfinite_tensors += 1
                continue
            squared_norm += float(torch.sum(detached.float().square()).cpu())
            max_abs = max(max_abs, float(detached.abs().max().cpu()))
        return {
            "tensors": tensors,
            "elements": elements,
            "nonfinite_tensors": nonfinite_tensors,
            "finite": nonfinite_tensors == 0,
            "l2_norm": math.sqrt(squared_norm),
            "max_abs": max_abs,
        }

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        if stage == "train":
            self._validate_training_batch(batch)
        output = self.model(batch["input_ids"], labels=batch["labels"], **self._model_inputs(batch))
        if output.loss is None:
            raise RuntimeError("model did not return a loss")
        supervised = batch["labels"][:, 1:].ne(-100).sum()
        if not bool(supervised):
            raise ValueError("batch has no assistant-token targets")
        batch_size = batch["input_ids"].shape[0]
        self.log(f"{stage}/loss", output.loss, on_step=stage == "train", on_epoch=True,
                 prog_bar=True, batch_size=batch_size, sync_dist=True)
        for name, value in (("tokens", batch["attention_mask"].sum()),
                            ("supervised_tokens", supervised)):
            self.log(f"{stage}/{name}", value.float(), on_step=stage == "train",
                     on_epoch=True, reduce_fx="sum", batch_size=batch_size, sync_dist=True)
        if stage == "train":
            self._pending_micro_batches.append({
                "loss": float(output.loss.detach().cpu()),
                "tokens": int(batch["attention_mask"].sum().detach().cpu()),
                "supervised_tokens": int(supervised.detach().cpu()),
                "segments": int(sum(
                    row[row >= 0].unique().numel() for row in batch["segment_ids"]
                )),
            })
        return output.loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def on_train_epoch_start(self) -> None:
        dataset = getattr(self.trainer.train_dataloader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Snapshot every update's accumulated, unscaled state before clipping."""
        update_step = self.trainer.global_step + 1
        if update_step % self.audit_every_steps:
            self._pending_micro_batches.clear()
            return
        micro_batches = self._pending_micro_batches
        self._pending_micro_batches = []
        if not micro_batches:
            raise RuntimeError("optimizer step has no recorded training micro-batches")
        supervised = sum(int(item["supervised_tokens"]) for item in micro_batches)
        weighted_loss = sum(
            float(item["loss"]) * int(item["supervised_tokens"])
            for item in micro_batches
        ) / supervised
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        gradients = self._tensor_summary(parameters, gradients=True)
        if not gradients["finite"]:
            raise RuntimeError("non-finite gradients detected before optimizer step")
        self._pending_audit = {
            "step": update_step,
            "epoch": self.current_epoch,
            "micro_batches": len(micro_batches),
            "loss_weighted_by_supervised_tokens": weighted_loss,
            "tokens": sum(int(item["tokens"]) for item in micro_batches),
            "supervised_tokens": supervised,
            "segments": sum(int(item["segments"]) for item in micro_batches),
            "learning_rates": {
                str(group.get("group_name", index)): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            "parameters": self._tensor_summary(parameters, gradients=False),
            "gradients_pre_clip": gradients,
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | int | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Apply Lightning-configured clipping and append the per-step audit row."""
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
        if self._pending_audit is None:
            return
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        gradients = self._tensor_summary(parameters, gradients=True)
        if not gradients["finite"]:
            raise RuntimeError("non-finite gradients detected after clipping")
        self._pending_audit["gradient_clip_value"] = gradient_clip_val
        self._pending_audit["gradient_clip_algorithm"] = gradient_clip_algorithm
        self._pending_audit["gradients_post_clip"] = gradients
        # These compact metrics are visible in both CSV and W&B. The matching
        # JSONL record has the complete per-group and health detail for review.
        self.log("audit/grad_norm_pre_clip", float(self._pending_audit["gradients_pre_clip"]["l2_norm"]), on_step=True, logger=True)
        self.log("audit/grad_norm_post_clip", float(gradients["l2_norm"]), on_step=True, logger=True)
        self.log("audit/grad_max_abs_post_clip", float(gradients["max_abs"]), on_step=True, logger=True)
        self.log("audit/parameter_max_abs", float(self._pending_audit["parameters"]["max_abs"]), on_step=True, logger=True)
        if self.trainer.is_global_zero:
            path = Path(self.trainer.default_root_dir) / "step_audit.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(self._pending_audit, sort_keys=True) + "\n")
        self._pending_audit = None

    def _lr_multiplier(self, scheduler_step: int) -> float:
        update = scheduler_step + 1
        if self.warmup_steps and update <= self.warmup_steps:
            return update / self.warmup_steps
        if self.lr_schedule == "constant":
            return 1.0
        assert self.total_steps is not None
        progress = min(1.0, (update - self.warmup_steps) /
                       (self.total_steps - self.warmup_steps))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress))

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = create_optimizer(self.model, self.optimizer_config)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_multiplier)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def _lightning_precision(precision: str) -> str:
    return {"fp32": "32-true", "bf16": "bf16-mixed", "fp16": "16-mixed"}[precision]


def build_trainer(config: TrainerConfig, *, accelerator: str = "auto") -> L.Trainer:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=config.output_dir / "checkpoints", filename="step-{step:08d}",
        every_n_train_steps=config.checkpoint_every_steps, save_last=True,
        save_top_k=-1, auto_insert_metric_name=False)
    loggers: list[Any] = [CSVLogger(config.output_dir, name="logs")]
    if config.wandb_mode != "disabled":
        loggers.append(WandbLogger(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            save_dir=config.output_dir,
            offline=config.wandb_mode == "offline",
            log_model=config.wandb_log_model,
        ))
    return L.Trainer(
        accelerator=accelerator, devices="auto", precision=_lightning_precision(config.precision),
        max_steps=config.total_steps if config.total_steps is not None else -1,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=config.max_grad_norm, gradient_clip_algorithm="norm",
        val_check_interval=config.validation_every_steps, check_val_every_n_epoch=None,
        log_every_n_steps=config.log_every_steps,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        logger=loggers,
        default_root_dir=config.output_dir, enable_checkpointing=True)


def run(config_path: str | Path, model_type: str, *, resume: str | Path | None = None) -> None:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    options = raw.get("models", {}).get(model_type)
    if options is None:
        raise ValueError(f"config has no model {model_type!r}")
    total_steps = int(raw["optimizer_updates"])
    keys = ("batch_size", "gradient_accumulation_steps", "precision",
            "gradient_checkpointing", "max_grad_norm", "seed", "log_every_steps",
            "checkpoint_every_steps", "validation_every_steps", "num_workers", "audit_every_steps",
            "lr_schedule", "warmup_steps", "min_lr_ratio", "wandb_project",
            "wandb_entity", "wandb_run_name", "wandb_mode", "wandb_log_model")
    config = TrainerConfig(**{key: raw[key] for key in keys if key in raw},
                           total_steps=total_steps,
                           output_dir=Path(raw["output_dir"]) / model_type)
    optimizer = OptimizerConfig(
        learning_rate=float(raw["learning_rate"]), weight_decay=float(raw.get("weight_decay", 0.1)),
        bit_learning_rate_multiplier=float(raw.get("bit_learning_rate_multiplier", 1.0)),
        bit_weight_decay=float(raw.get("bit_weight_decay", 0.0)))
    data = AmarkenDataModule(
        tokenizer_path=raw["tokenizer"], train_data=raw["train_data"],
        validation_data=raw["validation_data"], train_split=raw.get("train_split", "train"),
        validation_split=raw.get("validation_split", "validation"),
        train_token_budget=raw.get("train_token_budget"),
        validation_token_budget=raw.get("validation_token_budget"),
        sequence_length=int(raw["sequence_length"]), batch_size=config.batch_size,
        seed=config.seed, num_workers=config.num_workers)
    module = AmarkenLightningModule(
        create_model(create_config(model_type, **options)), optimizer,
        lr_schedule=config.lr_schedule, warmup_steps=config.warmup_steps,
        total_steps=config.total_steps, min_lr_ratio=config.min_lr_ratio,
        gradient_checkpointing=config.gradient_checkpointing,
        audit_every_steps=config.audit_every_steps)
    L.seed_everything(config.seed, workers=True)
    trainer = build_trainer(config, accelerator=raw.get("device", "auto"))
    trainer.logger.log_hyperparams({
        "experiment": raw,
        "trainer": {**asdict(config), "output_dir": str(config.output_dir)},
    })
    trainer.fit(
        module, datamodule=data, ckpt_path=None if resume is None else str(resume))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("dt", "bit", "glimmer"), required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(args.config, args.model, resume=args.resume)


if __name__ == "__main__":
    main()
