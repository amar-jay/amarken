"""Full-precision deep-thin control model."""

from .config import DTConfig
from .model import DTCausalLM

__all__ = ["DTConfig", "DTCausalLM"]
