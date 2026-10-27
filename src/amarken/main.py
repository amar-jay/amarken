from dataclasses import replace
from pathlib import Path

from amarken.config import Config
from amarken.dataset import build_dataloaders, build_dataset
from amarken.model import build_model, fsdp_modules, load_model, loss_fn, get_rl_hooks, build_metrics, validation_step
from amarken.trainer import Trainer
from amarken.utils.checkpoint import validate_initial_policy
from amarken.utils.distributed import launched, training_process
from amarken.utils.helpers import seed_everything


def run_train(config: Config, *, resume: Path | None = None):
    """Train the selected model and dataset without parsing command-line arguments."""
    if config.rl is not None and (launched() or config.strategy != "auto"):
        raise ValueError(
            "GRPO currently supports one device without torchrun, DDP or FSDP"
        )
    with training_process(
        config.device, strategy=config.strategy, precision=config.precision
    ) as device:
        return _train(replace(config, device=device), resume=resume)


def _train(config, *, resume):
    seed_everything(config.seed)
    rl_hooks = get_rl_hooks() if config.rl is not None else {}
    # Preserve FP32 initialization and cast frozen subtrees before GPU placement.
    options = (
        {"fine_tuning": config.fine_tuning} if config.fine_tuning is not None else {}
    )
    if config.rl is not None and config.rl.initial_checkpoint and resume is None:
        validate_initial_policy(config.rl.initial_checkpoint, config)
        model = load_model(
            config.rl.initial_checkpoint,
            model_id=config.model_id,
            device="cpu",
            precision="float32",
            **options,
        )
    else:
        model = build_model(
            model_id=config.model_id, device="cpu", precision="float32", **options
        )
    train_data, validation_data = build_dataset(config.dataset, seed=config.seed)
    train, validation = build_dataloaders(
        config.dataset,
        train_data,
        validation_data,
        batch_size=config.batch_size,
        tokenize=getattr(model, "tokenize", None),
    )
    if config.rl is not None:
        from amarken.rl_trainer import GRPOTrainer

        trainer = GRPOTrainer(
            model,
            train,
            config,
            validation_loader=validation,
            **rl_hooks,
            build_metrics=build_metrics,
            validation_step=validation_step,
        )
    else:
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
