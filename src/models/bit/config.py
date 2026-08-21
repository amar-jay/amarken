"""Configuration for Amarken-Bit, a sub-60M native ternary decoder."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BitConfig:
    """Static architecture/training-forward contract.

    The defaults transfer the released BitNet b1.58 2B4T block design while
    obeying Amarken's tokenizer, parameter ceiling, and FP-activation experiment.
    Ternary refers to every BitLinear projection; tied embeddings/head and norms
    remain floating point and are included honestly in artifact-byte accounting.
    """

    # Middle EN/TR tokenizer sweep point; tying limits this FP-heavy component to
    # one 6.144M matrix rather than separate input and output matrices.
    vocab_size: int = 12_000
    # Matched to DT/Glimmer so quality differences isolate ternary/block choices.
    hidden_size: int = 512
    # ReLU2-GLU still has three matrices. 1472 spends the remaining budget while
    # keeping the default model below 60M; it is 2.875x hidden vs BitNet's 2.7x.
    intermediate_size: int = 1_472
    # Matched 18-layer depth avoids granting extra sequential compute to Bit.
    num_hidden_layers: int = 18
    # 8x64 reconstructs the 512 residual width and preserves head diversity.
    num_attention_heads: int = 8
    # 4:1 GQA copies released BitNet (20Q/5KV) and reduces KV bytes/projections.
    num_key_value_heads: int = 2
    head_dim: int = 64
    # Context contract; RoPE is analytic, but full attention remains O(T^2).
    max_position_embeddings: int = 8_192
    # Released model value; slower rotations are preferable for longer contexts.
    rope_theta: float = 500_000.0
    # Released BitNet normalization epsilon, used for pre-norm and both SubLNs.
    rms_norm_eps: float = 1e-5
    # BitNet/modern decoder default. Nonzero remains an explicit regularization sweep.
    attention_dropout: float = 0.0
    # FP master-weight initialization used before each forward's fake quantization.
    initializer_range: float = 0.02
    # Amarken intentionally keeps activations FP; 8-bit activation QAT is a separate
    # experiment because otherwise weight and activation effects are confounded.
    quantize_activations: bool = False
    # Required project invariant and used by the released 2B configuration.
    tie_word_embeddings: bool = True
    # Numerical floor for absmean weight scale; prevents zero-matrix division by zero.
    quantization_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if min(self.vocab_size, self.intermediate_size, self.num_hidden_layers) < 1:
            raise ValueError("vocab, intermediate size, and layer count must be positive")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.quantize_activations:
            raise ValueError("Amarken-Bit currently specifies FP activations; use a separate activation-QAT branch")

    def to_dict(self) -> dict:
        return asdict(self)
