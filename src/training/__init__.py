"""Shared training and blocking correctness infrastructure.

The correctness runner stays in ``src.training.correctness`` so invoking it as a
module does not encounter an eager-import warning.
"""

from .data import PackedSequenceDataset, TokenizedExample
from .optimizer import OptimizerConfig, create_optimizer
from .trainer import Trainer, TrainerConfig, TrainerState

__all__ = [
    "OptimizerConfig",
    "PackedSequenceDataset",
    "TokenizedExample",
    "Trainer",
    "TrainerConfig",
    "TrainerState",
    "create_optimizer",
]
