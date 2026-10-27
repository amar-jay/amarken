from dataclasses import replace
from pathlib import Path

from amarken.dataset import build_dataloaders, build_dataset
from amarken.distributed import training_process
from amarken.model import build_model, fsdp_modules, loss_fn
from amarken.trainer import Trainer, TrainingConfig, seed_everything


def run_train(config: TrainingConfig, *, resume: Path | None = None):
    """Train the selected model and dataset without parsing command-line arguments."""
    with training_process(
        config.device, strategy=config.strategy, precision=config.precision
    ) as device:
        return _train(replace(config, device=device), resume=resume)


def _train(config, *, resume):
    seed_everything(config.seed)
    # Preserve FP32 initialization and cast frozen subtrees before GPU placement.
    model = build_model(model_id=config.model_id, device="cpu", precision="float32")
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
