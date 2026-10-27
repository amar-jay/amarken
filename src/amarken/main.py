from dataclasses import replace
from pathlib import Path

from amarken.dataset import build_dataloaders, build_dataset
from amarken.distributed import resolve_strategy, training_process
from amarken.model import build_model, fsdp_modules, loss_fn
from amarken.trainer import Trainer, TrainingConfig, seed_everything


def train_example(config: TrainingConfig, *, resume: Path | None = None):
    """Train the selected model and dataset without parsing command-line arguments."""
    with training_process(
        config.device, strategy=config.strategy, precision=config.precision
    ) as device:
        return _train(replace(config, device=device), resume=resume)


def _train(config, *, resume):
    seed_everything(config.seed)
    strategy = resolve_strategy(config.strategy, config.precision)
    model_device = "cpu" if strategy == "fsdp2" else config.device
    model = build_model(device=model_device, precision=config.precision)
    train_data, validation_data = build_dataset(
        seed=config.seed, train_size=config.train_size, val_size=config.val_size
    )
    train, validation = build_dataloaders(
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
