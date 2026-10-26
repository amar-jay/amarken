import torch


def loss_fn(model, batch, device):
    inputs, targets = (value.to(device) for value in batch)
    return torch.nn.functional.cross_entropy(model(inputs), targets)
