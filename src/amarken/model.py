"""Select your implementation here; training and inference share this import.

Switch base to example below to use the numeric example. Each module
exports build_model, load_model and predict with matching call signatures.
"""

from amarken.modules.base import build_model, load_model, predict

__all__ = ["build_model", "load_model", "predict"]
