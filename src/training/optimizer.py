"""Shared, fail-closed AdamW construction for every Amarken architecture.

This is deliberately not a Bit-only optimizer. DT and Glimmer receive ordinary
decay/no-decay groups. The Bit model gets one additional group for the floating-
point master weights owned by ``BitLinear``; quantization still happens in the
model forward pass through its straight-through estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from torch import nn
from torch.optim import AdamW

from src.models.common import AmarkenCausalLM


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    # These two settings are consulted only when the model actually contains
    # BitLinear modules. They cannot accidentally alter DT or Glimmer groups.
    bit_learning_rate_multiplier: float = 1.0
    bit_weight_decay: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.learning_rate,
            self.weight_decay,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.bit_learning_rate_multiplier,
            self.bit_weight_decay,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("optimizer values must be finite")
        if self.learning_rate <= 0 or self.epsilon <= 0:
            raise ValueError("learning rate and epsilon must be positive")
        if self.weight_decay < 0 or self.bit_weight_decay < 0:
            raise ValueError("weight decay cannot be negative")
        if not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1:
            raise ValueError("Adam betas must be in (0,1)")
        if self.bit_learning_rate_multiplier <= 0:
            raise ValueError("bit_learning_rate_multiplier must be positive")


def _ternary_master_ids(model: AmarkenCausalLM) -> set[int]:
    """Return only the latent FP weights that BitLinear quantizes in forward."""
    if model.config.model_type != "bit":
        return set()

    # Keep the Bit dependency behind the architecture check. Importing or
    # changing Bit internals therefore cannot affect a DT/Glimmer optimizer.
    from src.models.bit.model import BitLinear

    masters = {id(module.weight) for module in model.modules() if isinstance(module, BitLinear)}
    if not masters:
        raise RuntimeError("Bit model contains no BitLinear master weights")
    return masters


def parameter_groups(model: AmarkenCausalLM, config: OptimizerConfig) -> list[dict[str, Any]]:
    """Partition every trainable tensor exactly once or refuse to train."""
    modules = dict(model.named_modules())
    ternary_ids = _ternary_master_ids(model)
    grouped: dict[str, list[nn.Parameter]] = {
        "decay": [],
        "no_decay": [],
        "ternary_master": [],
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        module_name, _, local_name = name.rpartition(".")
        owner = modules[module_name]

        if id(parameter) in ternary_ids:
            grouped["ternary_master"].append(parameter)
        # AdamW decay is reserved for ordinary learned matrix/tensor weights.
        # Biases, normalization/scaling vectors, and embeddings are excluded.
        elif local_name == "bias" or parameter.ndim < 2 or isinstance(owner, nn.Embedding):
            grouped["no_decay"].append(parameter)
        else:
            grouped["decay"].append(parameter)

    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assigned = [id(parameter) for values in grouped.values() for parameter in values]
    # Compare identities, not numel totals: two missing tensors can have the same
    # combined size as a duplicated tensor and would evade a size-only check.
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise RuntimeError("optimizer groups must contain every trainable parameter exactly once")
    if set(id(parameter) for parameter in grouped["ternary_master"]) != ternary_ids:
        raise RuntimeError("BitLinear master-weight classification is incomplete")

    policies = {
        "decay": (config.learning_rate, config.weight_decay),
        "no_decay": (config.learning_rate, 0.0),
        "ternary_master": (
            config.learning_rate * config.bit_learning_rate_multiplier,
            config.bit_weight_decay,
        ),
    }
    result: list[dict[str, Any]] = []
    for group_name, parameters in grouped.items():
        if not parameters:
            continue
        learning_rate, weight_decay = policies[group_name]
        result.append({
            "params": parameters,
            "group_name": group_name,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        })
    return result


def create_optimizer(model: AmarkenCausalLM, config: OptimizerConfig) -> AdamW:
    """Create AdamW after the architecture-aware grouping audit succeeds."""
    return AdamW(
        parameter_groups(model, config),
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )
