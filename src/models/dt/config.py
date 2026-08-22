"""Configuration for the full-precision deep-thin control Transformer."""

from dataclasses import asdict, dataclass
from typing import ClassVar


@dataclass(frozen=True)
class DTConfig:
    """Conventional RMSNorm/RoPE/SwiGLU/GQA decoder used as the control arm."""

    vocab_size: int = 12_000
    hidden_size: int = 512
    intermediate_size: int = 1_344
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 64
    max_position_embeddings: int = 8_192
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if min(self.vocab_size, self.intermediate_size, self.num_hidden_layers) < 1:
            raise ValueError("vocab, intermediate size, and layer count must be positive")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0,1)")

    def to_dict(self) -> dict:
        return asdict(self)

    model_type: ClassVar[str] = "dt"
