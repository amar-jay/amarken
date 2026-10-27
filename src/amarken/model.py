"""Select your implementation here; training, inference and validation share this import.

Switch base to example below to use the numeric example. Each module
exports build/load, prediction, loss, validation and fsdp_modules hooks.
"""

from amarken.modules.example import (
    build_model,
    fsdp_modules,
    load_model,
    loss_fn,
    predict,
    validation_step,
)

__all__ = [
    "build_model",
    "fsdp_modules",
    "load_model",
    "loss_fn",
    "predict",
    "validation_step",
]
