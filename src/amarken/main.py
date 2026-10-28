from dataclasses import replace
from pathlib import Path

from amarken.config import Config
from amarken.dataset import build_dataloaders, build_dataset
from amarken.model import (
    build_model,
    fsdp_modules,
    load_model,
    loss_fn,
)
from amarken.trainers.supervised import Trainer
from amarken.utils.checkpoint import validate_initial_policy
from amarken.utils.distributed import launched, training_process
from amarken.utils.helpers import seed_everything


def run_train(config: Config, *, resume: Path | None = None):
    """Train the selected model and dataset without parsing command-line arguments."""
    if config.rl is not None and (launched() or config.strategy != "auto"):
        raise ValueError(
            "TRL adapters currently support one device without torchrun, DDP or FSDP"
        )
    with training_process(
        config.device, strategy=config.strategy, precision=config.precision
    ) as device:
        return _train(replace(config, device=device), resume=resume)


def _train(config, *, resume):
    seed_everything(config.seed)
    # Preserve FP32 initialization and cast frozen subtrees before GPU placement.
    options = (
        {"fine_tuning": config.fine_tuning} if config.fine_tuning is not None else {}
    )
    policy_checkpoint = (
        (resume if resume is not None else config.rl.initial_checkpoint)
        if config.rl is not None
        else None
    )
    if policy_checkpoint is not None:
        if Path(policy_checkpoint).is_dir():
            raise ValueError(
                "RL resume requires an amarken .pt checkpoint with complete policy weights"
            )
        if resume is None:
            validate_initial_policy(policy_checkpoint, config)
        model = load_model(
            policy_checkpoint,
            model_id=config.model_id,
            device="cpu",
            precision="float32",
            **options,
        )
    else:
        model = build_model(
            model_id=config.model_id, device="cpu", precision="float32", **options
        )
    if config.rl is not None:
        from amarken.trainers.rl import Trainer as RLTrainer

        trainer = RLTrainer.from_config(model, config)
    else:
        train_data, validation_data = build_dataset(config.dataset, seed=config.seed)
        train, validation = build_dataloaders(
            config.dataset,
            train_data,
            validation_data,
            batch_size=config.batch_size,
            tokenize=getattr(model, "tokenize", None),
        )
        trainer = Trainer(
            model,
            train,
            loss_fn,
            config,
            validation_loader=validation,
            fsdp_modules=fsdp_modules,
        )
    if resume is not None:
        trainer.resume(resume)
    history = trainer.fit()
    return trainer.checkpoint_prefix, history
