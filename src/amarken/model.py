"""Select your implementation here; training and inference share this import.

Switch example to base below to use the Hugging Face backbone. Each module
exports build_model, load_model and predict with matching call signatures.
"""

from amarken.modules.example import build_model, load_model, predict

__all__ = ["build_model", "load_model", "predict"]
