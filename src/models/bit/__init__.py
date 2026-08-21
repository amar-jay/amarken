"""Native ternary Amarken-Bit language model."""

from .config import BitConfig
from .model import ArtifactReport, BitCausalLM, BitLinear, pack_ternary
from ..common import CausalLMOutput

# Compatibility alias: new code should use CausalLMOutput for both architectures.
BitOutput = CausalLMOutput

__all__ = ["ArtifactReport", "BitCausalLM", "BitConfig", "BitLinear", "BitOutput", "pack_ternary"]
