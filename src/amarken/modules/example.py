"""Shared model definition for synthetic training and inference."""

from torch import nn


def build_example_model() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
