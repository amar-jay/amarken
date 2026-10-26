"""Synthetic training workflow, independent of command-line parsing."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from amarken.modules.example import build_example_model
from amarken.modules.loss_fn import loss_fn
from amarken.trainer import Trainer, TrainingConfig, seed_everything


def train_example(config: TrainingConfig, *, resume: Path | None = None):
    """Train the synthetic task without parsing command-line arguments."""
    seed_everything(config.seed)
    # Dedicated generator keeps dataset generation independent of training RNG.
    generator = torch.Generator().manual_seed(config.seed)
    inputs = torch.randn(160, 8, generator=generator)
    labels = (inputs[:, 0] + inputs[:, 1] > 0).long()
    train = DataLoader(
        TensorDataset(inputs[:128], labels[:128]),
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation = DataLoader(
        TensorDataset(inputs[128:], labels[128:]), batch_size=config.batch_size
    )
    model = build_example_model()
    trainer = Trainer(model, train, loss_fn, config, validation_loader=validation)
    if resume is not None:
        trainer.resume(resume)
    history = trainer.fit()
    return trainer.checkpoint_prefix, history
