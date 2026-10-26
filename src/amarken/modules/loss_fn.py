"""Example loss; accepts tensor inputs or a tokenized input dictionary."""

from collections.abc import Mapping

import torch


def loss_fn(model, batch, device):
    inputs, targets = batch
    if isinstance(inputs, Mapping):
        inputs = {key: value.to(device) for key, value in inputs.items()}
    else:
        inputs = inputs.to(device)
    return torch.nn.functional.cross_entropy(model(inputs), targets.to(device))
