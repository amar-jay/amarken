"""Full-precision deep-thin Transformer control for architecture tournaments."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import DTConfig
from ..common import AmarkenCausalLM, CausalLMOutput


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # FP32 reduction makes the control numerically stable under the exact same
        # BF16 autocast used by Glimmer and Bit.
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", frequencies, persistent=False)

    def forward(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        phases = torch.einsum("bt,d->btd", positions.float(), self.inv_freq.float())
        phases = torch.cat((phases, phases), dim=-1)
        return phases.cos().to(dtype), phases.sin().to(dtype)


def _rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, config: DTConfig):
        super().__init__()
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        self.config = config
        # Bias-free GQA is the conventional full-precision baseline. It deliberately
        # omits Glimmer's gate/QK norm/query scale so those features remain testable.
        self.q_proj = nn.Linear(config.hidden_size, q_width, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.o_proj = nn.Linear(q_width, config.hidden_size, bias=False)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], mask: Tensor | None) -> Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.config.num_attention_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.config.num_key_value_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.config.num_key_value_heads, -1).transpose(1, 2)
        q, k = _rope(q, *rope), _rope(k, *rope)
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=mask is None,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
        )
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, -1))


class SwiGLU(nn.Module):
    def __init__(self, config: DTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: DTConfig):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Attention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], mask: Tensor | None) -> Tensor:
        x = x + self.attention(self.input_norm(x), rope, mask)
        return x + self.mlp(self.post_attention_norm(x))


class DTCausalLM(AmarkenCausalLM[DTConfig]):
    config_type = DTConfig

    def __init__(self, config: DTConfig | None = None):
        super().__init__()
        config = DTConfig() if config is None else config
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_embedding = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _resource_accounting(self, sequence_length: int, element_bytes: int) -> tuple[int, int, int, int]:
        # Dense learned matrices all execute every token. Attention counts QK and
        # AV over the causal triangle; elementwise/norm/RoPE follow common policy.
        matrix_parameters = sum(module.weight.numel() for module in self.modules() if isinstance(module, nn.Linear))
        pairs = sequence_length * (sequence_length + 1) // 2
        attention_flops = self.config.num_hidden_layers * 4 * self.config.hidden_size * pairs
        linear_flops = 2 * sequence_length * matrix_parameters
        kv_elements = self.config.num_hidden_layers * 2 * self.config.num_key_value_heads * self.config.head_dim * sequence_length
        return linear_flops + attention_flops, kv_elements * element_bytes, self.parameter_count() * element_bytes, 0

    def _padding_mask(self, attention_mask: Tensor, batch: int, length: int) -> Tensor:
        if attention_mask.shape != (batch, length):
            raise ValueError("attention_mask must have shape [batch, sequence]")
        valid = attention_mask.bool()
        causal = torch.ones((length, length), device=valid.device, dtype=torch.bool).tril()
        allowed = causal.unsqueeze(0) & valid[:, None, :]
        # Padded queries receive a self-edge to prevent all-masked SDPA rows; their
        # labels remain -100, so this numerical safeguard adds no training target.
        allowed |= (~valid)[:, :, None] & torch.eye(length, device=valid.device, dtype=torch.bool).unsqueeze(0)
        return allowed.unsqueeze(1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, labels: Tensor | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        batch, length = input_ids.shape
        if attention_mask is None:
            positions = torch.arange(length, device=input_ids.device).expand(batch, -1)
            mask = None
        else:
            valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
            positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
            mask = self._padding_mask(valid, batch, length)
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
        return CausalLMOutput(logits=logits, loss=loss)
