"""Amarken-Glimmer language model."""

from .config import GlimmerConfig
from .model import GlimmerCausalLM
from ..common import CausalLMOutput

# Compatibility alias: new code should import the architecture-neutral output.
GlimmerOutput = CausalLMOutput

__all__ = ["GlimmerCausalLM", "GlimmerConfig", "GlimmerOutput"]
