"""Native ternary-weight Transformer inspired by BitNet b1.58 2B4T.

Training retains FP master parameters for the optimizer but every BitLinear
forward uses absmean-scaled {-1,0,+1} weights through a straight-through
estimator (STE). This is quantization-aware training from scratch, not PTQ.
Activations and normalization remain floating point by Amarken's experiment
definition. Efficient packed inference requires a ternary kernel; ordinary
PyTorch below proves training semantics, not ternary runtime speed.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import BitConfig


def _effective_ternary(weight: Tensor, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    """Return STE effective weight, integer trits, and scalar absmean scale."""
    # Paper equation: gamma=mean(abs(W)); round+clip(W/gamma) gives trits. FP32
    # statistics make thresholds deterministic under mixed-precision autocast.
    scale = weight.float().abs().mean().clamp_min(eps)
    trits = torch.round(weight.float() / scale).clamp(-1, 1)
    quantized = (trits * scale).to(weight.dtype)
    # Forward equals quantized; derivative w.r.t. the FP master weight is identity.
    # Detaching the complete correction intentionally ignores scale/round gradients.
    effective = weight + (quantized - weight).detach()
    return effective, trits.to(torch.int8), scale


def pack_ternary(trits: Tensor) -> tuple[Tensor, int]:
    """Pack four {-1,0,+1} values into each uint8; return bytes and pad count."""
    if not torch.all((trits >= -1) & (trits <= 1)):
        raise ValueError("trits must contain only -1, 0, or 1")
    # Code mapping -1->0, 0->1, +1->2 leaves code 3 invalid for corruption checks.
    codes = (trits.to(torch.int16).flatten() + 1).to(torch.uint8)
    padding = (-codes.numel()) % 4
    if padding:
        # Pad with zero-valued weights (code 1); shape metadata removes them on load.
        codes = F.pad(codes, (0, padding), value=1)
    codes = codes.view(-1, 4)
    packed = codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
    return packed, padding


class RMSNorm(nn.Module):
    """FP learned RMSNorm: no mean subtraction or bias, matching BitNet."""

    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # FP32 reduction prevents low-precision variance overflow/underflow.
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class BitLinear(nn.Module):
    """Bias-free linear projection with FP master and ternary effective weights."""

    def __init__(self, in_features: int, out_features: int, eps: float):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_eps = eps
        # FP masters are necessary during optimization; only exported inference
        # artifacts replace this matrix with packed trits plus one FP scale.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def quantized(self) -> tuple[Tensor, Tensor]:
        """Detached integer trits and scale for inspection/export."""
        _, trits, scale = _effective_ternary(self.weight, self.quantization_eps)
        return trits.detach(), scale.detach()

    def forward(self, x: Tensor) -> Tensor:
        effective, _, _ = _effective_ternary(self.weight, self.quantization_eps)
        return F.linear(x, effective)


class RotaryEmbedding(nn.Module):
    """LLaMA half-split RoPE shared by every attention layer."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        phases = torch.einsum("bt,d->btd", positions.float(), self.inv_freq.float())
        phases = torch.cat((phases, phases), dim=-1)
        return phases.cos().to(dtype), phases.sin().to(dtype)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


class BitAttention(nn.Module):
    """Full causal GQA; all four learned projections are ternary BitLinear."""

    def __init__(self, config: BitConfig):
        super().__init__()
        self.config = config
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        make = lambda input_width, output_width: BitLinear(input_width, output_width, config.quantization_eps)
        self.q_proj = make(config.hidden_size, q_width)
        self.k_proj = make(config.hidden_size, kv_width)
        self.v_proj = make(config.hidden_size, kv_width)
        # Released SubLN normalizes concatenated attention features before o_proj;
        # this stabilizes the distribution entering a heavily quantized matrix.
        self.attn_sub_norm = RMSNorm(q_width, config.rms_norm_eps)
        self.o_proj = make(q_width, config.hidden_size)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], mask: Tensor | None) -> Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.config.num_attention_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.config.num_key_value_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.config.num_key_value_heads, -1).transpose(1, 2)
        q, k = _apply_rope(q, *rope), _apply_rope(k, *rope)
        # Repeat is the portable training path. Projection/KV-cache dimensions still
        # retain GQA savings; optimized inference should use a native grouped kernel.
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=mask is None,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(self.attn_sub_norm(attended))


class ReLU2GLU(nn.Module):
    """Released FFN: down(SubLN(ReLU(gate)^2 * up)), all projections ternary."""

    def __init__(self, config: BitConfig):
        super().__init__()
        make = lambda input_width, output_width: BitLinear(input_width, output_width, config.quantization_eps)
        self.gate_proj = make(config.hidden_size, config.intermediate_size)
        self.up_proj = make(config.hidden_size, config.intermediate_size)
        self.ffn_sub_norm = RMSNorm(config.intermediate_size, config.rms_norm_eps)
        self.down_proj = make(config.intermediate_size, config.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        # ReLU2 supplies activation sparsity and avoids SiLU's transcendental cost;
        # multiplication by an independent up branch retains GLU expressivity.
        hidden = F.relu(self.gate_proj(x)).square() * self.up_proj(x)
        return self.down_proj(self.ffn_sub_norm(hidden))


class BitDecoderLayer(nn.Module):
    """Pre-norm sequential attention/FFN residual block with internal SubLNs."""

    def __init__(self, config: BitConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = BitAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = ReLU2GLU(config)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], mask: Tensor | None) -> Tensor:
        x = x + self.attention(self.input_norm(x), rope, mask)
        return x + self.mlp(self.post_attention_norm(x))


@dataclass(frozen=True)
class ArtifactReport:
    total_parameters: int
    ternary_parameters: int
    floating_parameters: int
    theoretical_bytes: int
    packed_2bit_bytes: int
    training_master_bytes_fp32: int


@dataclass
class BitOutput:
    logits: Tensor
    loss: Tensor | None = None


class BitCausalLM(nn.Module):
    """Amarken-sized BitNet decoder trained from scratch with online ternarization."""

    def __init__(self, config: BitConfig | None = None):
        super().__init__()
        config = BitConfig() if config is None else config
        self.config = config
        # Released BitNet keeps embeddings/output FP; tying makes this one matrix.
        # Quantizing it should be a separate ablation because lookup errors and
        # projection errors otherwise obscure the contribution of BitLinear blocks.
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_embedding = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.layers = nn.ModuleList(BitDecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (BitLinear, nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def parameter_count(self) -> int:
        # PyTorch deduplicates the tied embedding/head Parameter automatically.
        return sum(parameter.numel() for parameter in self.parameters())

    def _padding_causal_mask(self, attention_mask: Tensor, batch: int, length: int) -> Tensor:
        if attention_mask.shape != (batch, length):
            raise ValueError("attention_mask must have shape [batch, sequence]")
        valid = attention_mask.to(dtype=torch.bool)
        causal = torch.ones((length, length), device=valid.device, dtype=torch.bool).tril()
        allowed = causal.unsqueeze(0) & valid[:, None, :]
        # Restore padded-query self edges after key masking: avoids all-masked rows
        # and backend-dependent NaNs while ignored padded labels carry no loss.
        allowed |= (~valid)[:, :, None] & torch.eye(length, device=valid.device, dtype=torch.bool).unsqueeze(0)
        return allowed.unsqueeze(1)  # [B,1,T,T], boolean True means participate in SDPA.

    def artifact_report(self, floating_bytes: int = 2) -> ArtifactReport:
        """Estimate deployable weights; excludes tokenizer/container alignment."""
        ternary = sum(module.weight.numel() for module in self.modules() if isinstance(module, BitLinear))
        total = self.parameter_count()
        floating = total - ternary
        scales = sum(1 for module in self.modules() if isinstance(module, BitLinear))
        # log2(3) is information-theoretic, while common kernels pack four 2-bit
        # trits per byte. Each BitLinear also needs one FP32 dequantization scale.
        theoretical = math.ceil(ternary * math.log2(3) / 8) + floating * floating_bytes + scales * 4
        packed = math.ceil(ternary / 4) + floating * floating_bytes + scales * 4
        return ArtifactReport(total, ternary, floating, theoretical, packed, total * 4)

    def export_ternary(self) -> dict[str, dict[str, Tensor | tuple[int, ...] | int]]:
        """Create an in-memory, lossless 2-bit projection payload for inference tooling."""
        result: dict[str, dict[str, Tensor | tuple[int, ...] | int]] = {}
        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                trits, scale = module.quantized()
                packed, padding = pack_ternary(trits.cpu())
                result[name] = {"packed": packed, "scale": scale.cpu(), "shape": tuple(trits.shape), "padding": padding}
        return result

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> BitOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, length = input_ids.shape
        if length > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        if attention_mask is None:
            # None lets SDPA use its native causal path/Flash kernel without a dense mask.
            positions = torch.arange(length, device=input_ids.device).expand(batch, -1)
            mask = None
        else:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
            # cumsum makes the first real token position zero under left padding;
            # padded slots clamp to zero and are neutralized by the attention mask.
            positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
            mask = self._padding_causal_mask(valid, batch, length)
        hidden = self.token_embedding(input_ids)
        rope = self.rotary_embedding(positions, hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, rope, mask)
        logits = self.lm_head(self.final_norm(hidden)).float()
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return BitOutput(logits, loss)
