"""Configuration for the sub-60M Amarken-Glimmer decoder."""

from dataclasses import asdict, dataclass
import math
from typing import ClassVar


@dataclass(frozen=True)
class GlimmerConfig:
    """Architecture configuration.

    Defaults target the 12k tokenizer candidate and stay below the project's
    60M parameter ceiling. Full/NoPE layers are counted backward from the last
    layer, matching the released Muse Glimmer implementation even when depth is
    not divisible by four.
    """

    # 12k is the middle EN/TR tokenizer candidate: materially less embedding
    # cost than 16k but less fragmentation than 8k; tied embedding cost=6.144M.
    vocab_size: int = 12_000
    # 512 preserves the project's deep-thin baseline and gives 8x64 Q heads;
    # width dominates every dense projection, so increasing it breaks 60M fast.
    hidden_size: int = 512
    # 2.625x hidden; selected so 3 SwiGLU matrices + gated attention + embeddings
    # total 58.676M, leaving ~1.3M under the hard ceiling for small additions.
    intermediate_size: int = 1_344
    # Depth is favored over width for sequential composition; 18 matches DT and
    # yields five periodic global layers when the pattern is anchored at output.
    num_hidden_layers: int = 18
    # Eight query heads retain multiple attention subspaces at 512d.
    num_attention_heads: int = 8
    # One KV head gives 8:1 GQA: minimum KV/cache bytes and an explicit sweep
    # endpoint; 2 KV heads is the 4:1 control, while Muse itself uses 16:1.
    num_key_value_heads: int = 1
    # 64 is hidden/heads, even for paired RoPE dimensions, and conventional for
    # this scale; it is explicit so invalid width/head combinations fail early.
    head_dim: int = 64
    # Training/inference contract only (RoPE itself is analytic, not a table);
    # 8192 is practical for a 60M model while allowing long-context experiments.
    max_position_embeddings: int = 8_192
    # Middle of required 256/512/1024 sweep: local layers cost O(n*w), and 512
    # supplies exact recent order while periodic global layers transport content.
    sliding_window: int = 512
    # Copied from Muse: slow angular rotation reduces long-distance RoPE phase
    # distortion; only local layers consume RoPE, but matching isolates scaling.
    rope_theta: float = 500_000.0
    # Muse's post-QK-norm inverse-temperature initializer; this implementation
    # makes it independently learnable per Q head rather than a frozen scalar.
    qk_scale_factor: float = 3.87
    # Muse pre/final/QK normalization epsilon: enough BF16/FP16 stability without
    # noticeably biasing normal-variance vectors.
    rms_norm_eps: float = 1e-5
    # Muse post-branch epsilon: branch outputs can be tiny at initialization, so
    # the smaller epsilon avoids suppressing them before residual addition.
    post_norm_eps: float = 1e-8
    # Zero is the released architecture/default and maximizes data efficiency;
    # regularization should be tested explicitly rather than silently imposed.
    attention_dropout: float = 0.0
    # Standard small-transformer Gaussian init, retained for matched experiments.
    initializer_range: float = 0.02
    # Muse tanh cap prevents extreme logits/gradients; <=0 intentionally disables.
    final_logit_softcap: float = 20.0
    # None derives Muse's width-aware 1/sqrt(hidden/256); explicit value supports
    # ablation/checkpoint compatibility without changing the architecture code.
    output_multiplier: float | None = None
    # Required by project: halves vocab parameters versus a separate LM head and
    # couples input/output lexical geometry; parameters are counted only once.
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        # Q concatenation must exactly reconstruct the residual stream width.
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        # Integer grouping is required to replicate each KV head equally.
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        # rotate_half splits each head into equal real/imaginary coordinate sets.
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        # Zero depth has no decoder; zero window would mask every key including self.
        if self.num_hidden_layers < 1 or self.sliding_window < 1:
            raise ValueError("num_hidden_layers and sliding_window must be positive")
        # PyTorch dropout probabilities are defined on this half-open interval.
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")

    @property
    def layer_types(self) -> tuple[str, ...]:
        # Count backward so the final block is always global/NoPE. At depths not
        # divisible by four this preserves a global information aggregation step
        # immediately before final norm/head, exactly as HF's Muse config does.
        return tuple(
            "full_attention"
            if (self.num_hidden_layers - 1 - index) % 4 == 0
            else "sliding_attention"
            for index in range(self.num_hidden_layers)
        )

    @property
    def logit_multiplier(self) -> float:
        # Explicit overrides are useful for ablation or importing a checkpoint.
        if self.output_multiplier is not None:
            return self.output_multiplier
        # Muse formula normalizes output scale across widths; at 512 this is 1/sqrt(2).
        return 1.0 / math.sqrt(self.hidden_size / 256.0)

    def to_dict(self) -> dict:
        # Shared config serialization must contain constructor fields only so the
        # dictionary round-trips through GlimmerConfig(**data). Derived layer_types
        # and logit_multiplier remain deterministic properties of these fields.
        return asdict(self)
    model_type: ClassVar[str] = "glimmer"
