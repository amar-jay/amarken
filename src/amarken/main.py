"""Synthetic classification example: python -m example.cli."""

import argparse
import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from amarken.modules.loss_fn import loss_fn
from amarken.trainer import Trainer, TrainingConfig, seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training-example.json")
    parser.add_argument("--resume", help="Local checkpoint file to resume")
    args = parser.parse_args(argv)
    with open(args.config) as file:
        config = TrainingConfig(**json.load(file))
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
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)
    )
    trainer = Trainer(model, train, loss_fn, config, validation_loader=validation)
    if args.resume:
        trainer.resume(args.resume)
    for metrics in trainer.fit():
        print(json.dumps(metrics))


if __name__ == "__main__":
    main()
