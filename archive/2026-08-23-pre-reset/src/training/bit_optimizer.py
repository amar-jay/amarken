"""Comparable AdamW parameter grouping with an explicit Bit-specific branch. (BitNet-like approach)"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch.optim import AdamW

from src.models.bit.model import BitLinear
from src.models.common import AmarkenCausalLM


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    # Ternary masters can need a smaller update scale because crossing an absmean
    # threshold changes the effective forward weight discontinuously under the STE.
    bit_learning_rate_multiplier: float = 1.0
    bit_weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.epsilon <= 0:
            raise ValueError("learning rate and epsilon must be positive")
        if not 0 <= self.weight_decay or not 0 <= self.bit_weight_decay:
            raise ValueError("weight decay cannot be negative")
        if not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1:
            raise ValueError("Adam betas must be in (0,1)")
        if self.bit_learning_rate_multiplier <= 0:
            raise ValueError("bit_learning_rate_multiplier must be positive")


def parameter_groups(model: AmarkenCausalLM, config: OptimizerConfig) -> list[dict]:
    """Partition every trainable parameter exactly once with auditable group names."""
    module_by_parameter = {}
    for module_name, module in model.named_modules():
        for local_name, parameter in module.named_parameters(recurse=False):
            module_by_parameter[id(parameter)] = (module_name, local_name, module)
    grouped: dict[str, list] = {"decay": [], "no_decay": [], "ternary_master": []}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        _, local_name, owner = module_by_parameter[id(parameter)]
        if isinstance(owner, BitLinear):
            grouped["ternary_master"].append(parameter)
        # Biases, one-dimensional normalization scales and embeddings are not
        # decayed; matrix weights receive ordinary AdamW decoupled decay.
        elif (
            local_name == "bias"
            or parameter.ndim < 2
            or isinstance(owner, nn.Embedding)
            or name == "token_embedding.weight"
        ):
            grouped["no_decay"].append(parameter)
        else:
            grouped["decay"].append(parameter)
    if sum(
        parameter.numel() for values in grouped.values() for parameter in values
    ) != model.parameter_count(True):
        raise RuntimeError(
            "optimizer grouping did not cover each trainable parameter exactly once"
        )
    result = []
    if grouped["decay"]:
        result.append(
            {
                "params": grouped["decay"],
                "group_name": "decay",
                "weight_decay": config.weight_decay,
                "lr": config.learning_rate,
                "base_lr": config.learning_rate,
            }
        )
    if grouped["no_decay"]:
        result.append(
            {
                "params": grouped["no_decay"],
                "group_name": "no_decay",
                "weight_decay": 0.0,
                "lr": config.learning_rate,
                "base_lr": config.learning_rate,
            }
        )
    if grouped["ternary_master"]:
        result.append(
            {
                "params": grouped["ternary_master"],
                "group_name": "ternary_master",
                "weight_decay": config.bit_weight_decay,
                "lr": config.learning_rate * config.bit_learning_rate_multiplier,
                "base_lr": config.learning_rate * config.bit_learning_rate_multiplier,
            }
        )
    return result


def create_optimizer(model: AmarkenCausalLM, config: OptimizerConfig) -> AdamW:
    return AdamW(
        parameter_groups(model, config),
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )
