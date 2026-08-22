"""Architecture-neutral gradient health and Bit ternary observability."""

from __future__ import annotations

import math

import torch

from src.models.bit.model import BitLinear
from src.models.common import AmarkenCausalLM


def gradient_health(model: AmarkenCausalLM) -> dict[str, float]:
    total = finite = zeros = 0
    squared_norm = maximum = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        total += gradient.numel()
        finite_mask = torch.isfinite(gradient)
        finite += int(finite_mask.sum())
        zeros += int((gradient == 0).sum())
        if finite_mask.any():
            values = gradient[finite_mask]
            squared_norm += float(values.square().sum())
            maximum = max(maximum, float(values.abs().max()))
    return {
        "grad_norm": math.sqrt(squared_norm),
        "grad_abs_max": maximum,
        "grad_finite_fraction": finite / total if total else 1.0,
        "grad_zero_fraction": zeros / total if total else 1.0,
    }


@torch.no_grad()
def ternary_statistics(model: AmarkenCausalLM) -> dict[str, float]:
    modules = [module for module in model.modules() if isinstance(module, BitLinear)]
    if not modules:
        return {}
    trit_counts = {-1: 0, 0: 0, 1: 0}
    scales = []
    ternary_grad_total = ternary_grad_finite = ternary_grad_zero = 0
    for module in modules:
        trits, scale = module.quantized()
        scales.append(scale.float())
        for value in trit_counts:
            trit_counts[value] += int((trits == value).sum())
        if module.weight.grad is not None:
            gradient = module.weight.grad.detach()
            ternary_grad_total += gradient.numel()
            ternary_grad_finite += int(torch.isfinite(gradient).sum())
            ternary_grad_zero += int((gradient == 0).sum())
    total = sum(trit_counts.values())
    scale_tensor = torch.stack(scales)
    return {
        "ternary_zero_fraction": trit_counts[0] / total,
        "ternary_negative_fraction": trit_counts[-1] / total,
        "ternary_positive_fraction": trit_counts[1] / total,
        "ternary_scale_min": float(scale_tensor.min()),
        "ternary_scale_mean": float(scale_tensor.mean()),
        "ternary_scale_max": float(scale_tensor.max()),
        "ternary_scale_std": float(scale_tensor.std(unbiased=False)),
        "ternary_grad_finite_fraction": ternary_grad_finite / ternary_grad_total if ternary_grad_total else 1.0,
        "ternary_grad_zero_fraction": ternary_grad_zero / ternary_grad_total if ternary_grad_total else 1.0,
    }
