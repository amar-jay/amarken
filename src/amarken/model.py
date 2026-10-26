"""Select your implementation here; training, inference and validation share this import.

Switch base to example below to use the numeric example. Each module
exports build_model, load_model, predict, loss_fn and validation_step.
"""

from amarken.modules.example import (
    build_model,
    load_model,
    loss_fn,
    predict,
    validation_step,
)

__all__ = ["build_model", "load_model", "loss_fn", "predict", "validation_step"]
