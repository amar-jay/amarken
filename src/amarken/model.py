"""Select your implementation here; training and inference share this import.

Switch base to example below to use the numeric example. Each module
exports build_model, load_model, predict and loss_fn with matching signatures.
"""

from amarken.modules.base import build_model, load_model, loss_fn, predict

__all__ = ["build_model", "load_model", "loss_fn", "predict"]
