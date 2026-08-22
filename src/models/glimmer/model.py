"""A compact, trainable implementation of the Muse Glimmer text architecture.

This is an Amarken-sized decoder, not a weight-compatible reproduction of the
30B multimodal checkpoint. It deliberately uses plain PyTorch so architecture
experiments do not depend on a particular Transformers release.
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import GlimmerConfig
from ..common import AmarkenCausalLM, CausalLMOutput, packed_positions


class RMSNorm(nn.Module):
    """Scale-only RMS normalization; no mean subtraction or additive bias.

    RMSNorm is cheaper than LayerNorm and preserves the direction/mean signal.
    ``scale=False`` implements Muse's parameter-free embedding and per-head QK
    norms; ``scale=True`` implements the learned final normalization.
    """

    def __init__(self, dim: int | None = None, eps: float = 1e-6, scale: bool = True):
        super().__init__()
        self.eps = eps
        # Omit the tensor entirely for scaleless norms: no hidden parameters and
        # exact unit-RMS output rather than a nominally frozen vector of ones.
        self.weight = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x: Tensor) -> Tensor:
        # Accumulate variance/rsqrt in FP32 to avoid FP16/BF16 overflow, underflow,
        # and compiler-dependent reductions; cast back to preserve mixed precision.
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            normalized = normalized * self.weight.float()
        return normalized.to(x.dtype)


class CenteredRMSNorm(nn.Module):
    """Muse branch norm parameterized as ``1 + weight`` initialized at identity.

    Zero-centered parameters improve low-precision optimization/weight decay
    behavior while representing the same scale family as ordinary RMSNorm.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Multiplication occurs in FP32 before the final cast, matching the released
        # Muse implementation and avoiding a subtly different low-precision graph.
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Default, non-scaled RoPE frequencies shared by all local layers."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        # One frequency per coordinate pair; larger theta makes high dimensions
        # rotate more slowly and extends distinguishable relative distances.
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        # Derived constant is device-movable but excluded from checkpoints because
        # theta/head_dim reconstruct it exactly and duplicate bytes are wasteful.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        # [B,T] outer [D/2] -> per-token phases [B,T,D/2]. FP32 phase math is
        # important at large positions; duplicate halves matches rotate_half layout.
        frequencies = torch.einsum("bt,d->btd", positions.float(), self.inv_freq.float())
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    # Implements multiplication by i on paired feature halves: (a,b)->(-b,a).
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # Insert the head axis so one position rotation broadcasts across every head;
    # x is [B,H,T,D], while cached cos/sin are [B,T,D].
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return x * cos + _rotate_half(x) * sin


class GatedGroupedQueryAttention(nn.Module):
    """Muse-style causal attention with local/full modes, GQA, QK norm and gate."""

    def __init__(self, config: GlimmerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # Type is immutable per layer and controls both receptive field and RoPE.
        self.layer_type = config.layer_types[layer_idx]
        # Q retains Hq heads; K/V store Hkv heads. Independent dimensions are the
        # source of GQA's parameter and future KV-cache savings.
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        # Bias-free projections match Muse and avoid redundant offsets around norms.
        self.q_proj = nn.Linear(config.hidden_size, q_width, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_width, bias=False)
        # Gate is computed from the same normalized branch input and modulates each
        # concatenated head feature before mixing by o_proj; sigmoid bounds it 0..1.
        self.gate_proj = (
            nn.Linear(config.hidden_size, q_width, bias=False) if config.use_attention_gate else None
        )
        self.o_proj = nn.Linear(q_width, config.hidden_size, bias=False)
        # One parameter-free norm is safe to reuse: RMSNorm has no running state;
        # it operates on the last dimension, therefore independently per head.
        self.qk_norm = RMSNorm(eps=config.rms_norm_eps, scale=False) if config.use_qk_norm else None
        # Per-head learnable inverse temperature; initialized from Muse Glimmer.
        self.query_scale = (
            nn.Parameter(torch.full((config.num_attention_heads,), config.qk_scale_factor))
            if config.use_qk_norm else None
        )

    def forward(
        self,
        hidden_states: Tensor,
        rope: tuple[Tensor, Tensor] | None,
        additive_mask: Tensor,
    ) -> Tensor:
        batch, length, _ = hidden_states.shape
        # Projection [B,T,H*D] -> [B,H,T,D], the layout expected by torch SDPA.
        q = self.q_proj(hidden_states).view(batch, length, self.config.num_attention_heads, -1).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, length, self.config.num_key_value_heads, -1).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, length, self.config.num_key_value_heads, -1).transpose(1, 2)
        # Unit-RMS Q/K stabilizes dot products; learned per-Q-head scale acts as an
        # inverse softmax temperature in addition to SDPA's standard 1/sqrt(D).
        if self.qk_norm is not None:
            q = self.qk_norm(q) * self.query_scale.view(1, -1, 1, 1)
            k = self.qk_norm(k)
        if rope is not None:
            # Local layers encode relative order/distance; global layers pass None
            # and are deliberately position-free to avoid long-distance RoPE decay.
            q = _apply_rope(q, *rope)
            k = _apply_rope(k, *rope)
        # Materialization is a portable eager implementation of GQA. It does not
        # reduce this training call's temporary K/V tensor, but projections and a
        # future compact cache retain the architectural Hkv savings.
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        # Mask geometry/padding is invariant across layers, so the caller passes one
        # of two stack-shared masks. Explicit mask and is_causal are mutually exclusive.
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
        )
        # Rejoin heads, apply bounded content-dependent gate, then mix/project back
        # to residual width. Gate-before-o_proj exactly matches Muse's operation order.
        output = output.transpose(1, 2).reshape(batch, length, -1)
        if self.gate_proj is not None:
            output = output * torch.sigmoid(self.gate_proj(hidden_states))
        return self.o_proj(output)


class SwiGLU(nn.Module):
    """Bias-free SwiGLU FFN: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: GlimmerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # Multiplicative gating offers more expressivity per parameter than a plain
        # two-matrix FFN; three matrices are included in all parameter accounting.
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GlimmerDecoderLayer(nn.Module):
    """Parallel-width sequential attention/FFN residual block with sandwich norms."""

    def __init__(self, config: GlimmerConfig, layer_idx: int):
        super().__init__()
        self.attention = GatedGroupedQueryAttention(config, layer_idx)
        self.mlp = SwiGLU(config)
        # Muse uses four norms, not conventional two-norm pre-norm: each branch is
        # normalized before computation and again before entering the residual.
        self.input_norm = CenteredRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = CenteredRMSNorm(config.hidden_size, config.post_norm_eps)
        self.pre_ffn_norm = CenteredRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_ffn_norm = CenteredRMSNorm(config.hidden_size, config.post_norm_eps)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None, additive_mask: Tensor) -> Tensor:
        # Sequential residuals (attention first, FFN sees updated state). Post norms
        # bound each residual update without normalizing/bypassing the residual path.
        x = x + self.post_attention_norm(self.attention(self.input_norm(x), rope, additive_mask))
        return x + self.post_ffn_norm(self.mlp(self.pre_ffn_norm(x)))


class GlimmerCausalLM(AmarkenCausalLM[GlimmerConfig]):
    """Text-only Amarken-Glimmer decoder for scratch training and evaluation."""

    config_type = GlimmerConfig

    def __init__(self, config: GlimmerConfig | None = None):
        super().__init__()
        # Construct defaults per instance. GlimmerConfig is currently frozen, but
        # avoiding a definition-time singleton remains safe if mutable fields are
        # added later and prevents surprising identity coupling between models.
        config = GlimmerConfig() if config is None else config
        self.config = config
        self.gradient_checkpointing = False
        # No positional embedding table: token embeddings carry content only and
        # RoPE is applied inside local attention. Padding row is tokenizer-dependent.
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # Muse normalizes embeddings without learned scale, controlling the input
        # magnitude while avoiding another hidden-size parameter vector.
        self.embedding_norm = RMSNorm(eps=config.rms_norm_eps, scale=False)
        # One RoPE generator is shared because all local layers use identical theta.
        self.rotary_embedding = RotaryEmbedding(config.head_dim, config.rope_theta)
        # Unique blocks (no recurrent sharing): logical depth equals parameter depth,
        # making this model a clean Glimmer-vs-DT architectural comparison.
        self.layers = nn.ModuleList(GlimmerDecoderLayer(config, i) for i in range(config.num_hidden_layers))
        # Learned final RMS scale prepares a consistent representation for logits.
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # Bias-free vocabulary projection; tying below aliases its weight object.
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize before tying so embedding and head are each initialized once;
        # the head's discarded independent initialization has no semantic effect.
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            # Assignment aliases storage (not a copy), ensuring optimizer updates and
            # parameter_count see one matrix and input/output lexical spaces coincide.
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        # Only learned affine/lookup matrices receive Gaussian init; norm scales use
        # their purposeful one/zero initialization and derived buffers are untouched.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _resource_accounting(self, sequence_length: int, element_bytes: int) -> tuple[int, int, int, int]:
        # Linear FLOPs cover every layer projection plus the tied output projection;
        # embedding lookup itself is indexing and therefore excluded.
        layer_linear_parameters = sum(
            module.weight.numel() for layer in self.layers for module in layer.modules() if isinstance(module, nn.Linear)
        )
        linear_flops = 2 * sequence_length * (
            layer_linear_parameters + self.config.hidden_size * self.config.vocab_size
        )

        def causal_pairs(tokens: int, window: int | None = None) -> int:
            if window is None or tokens <= window:
                return tokens * (tokens + 1) // 2
            return window * (window + 1) // 2 + (tokens - window) * window

        attention_flops = 0
        cached_kv_elements = 0
        for layer_type in self.config.layer_types:
            keys = sequence_length if layer_type == "full_attention" else min(sequence_length, self.config.sliding_window)
            pairs = causal_pairs(sequence_length, None if layer_type == "full_attention" else self.config.sliding_window)
            # QK and AV each cost ~2 FLOPs per hidden-dimension MAC => 4*H per pair.
            attention_flops += 4 * self.config.hidden_size * pairs
            cached_kv_elements += 2 * self.config.num_key_value_heads * self.config.head_dim * keys
        total = self.parameter_count()
        return linear_flops + attention_flops, cached_kv_elements * element_bytes, total * element_bytes, 0

    def _attention_masks(
        self,
        attention_mask: Tensor | None,
        batch: int,
        length: int,
        device: torch.device,
        segment_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Build the two immutable stack masks once per forward.

        Padding removes key columns. Padded query rows receive a self-edge after
        that removal so no SDPA backend ever sees an undefined all-``-inf`` row.
        """
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            valid = attention_mask.to(device=device, dtype=torch.bool)
        else:
            # Singleton batch broadcasts over examples without B copies when every
            # token is valid; supplied padding requires genuinely per-example masks.
            valid = torch.ones((1, length), device=device, dtype=torch.bool)

        # row=query, col=key. Equality preserves self-attention; strict local bound
        # exposes exactly W keys q-W+1..q, naturally clipped at sequence start.
        row = torch.arange(length, device=device)[:, None]
        col = torch.arange(length, device=device)[None, :]
        causal = col <= row
        local = causal & (col > row - self.config.sliding_window)
        identity = torch.eye(length, device=device, dtype=torch.bool)

        masks: dict[str, Tensor] = {}
        for layer_type, geometry in (
            ("full_attention", causal),
            ("sliding_attention", local),
        ):
            # Apply key padding first, then restore only padded-query diagonals. The
            # reverse order would immediately erase the safety edge with key masking.
            allowed = geometry.unsqueeze(0) & valid[:, None, :]
            if segment_ids is not None:
                if segment_ids.shape != (batch, length):
                    raise ValueError("segment_ids must have shape [batch, sequence]")
                allowed &= segment_ids[:, :, None] == segment_ids[:, None, :]
            allowed |= (~valid)[:, :, None] & identity.unsqueeze(0)
            # FP32 additive -inf composes exactly with attention scores. Shape
            # [B|1,1,T,T] broadcasts over query heads without materializing H copies.
            additive = torch.zeros(allowed.shape, device=device, dtype=torch.float32)
            additive.masked_fill_(~allowed, float("-inf"))
            masks[layer_type] = additive.unsqueeze(1)
        return masks

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        segment_ids: Tensor | None = None,
    ) -> CausalLMOutput:
        # Batch-major integer token IDs are the sole current input interface; direct
        # embeddings/cache support can be added without changing decoder blocks.
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        # Hard guard documents the evaluated context contract and catches accidental
        # unbounded O(T^2) allocation in periodic global layers.
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        batch, length = input_ids.shape
        if attention_mask is None:
            if segment_ids is not None:
                raise ValueError("segment_ids require attention_mask")
            positions = torch.arange(length, device=input_ids.device).expand(batch, -1)
        else:
            # Match Bit/common semantics: real tokens start at RoPE position zero
            # under left padding instead of shifting with batch padding length.
            valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
            positions = packed_positions(valid, segment_ids)
        hidden = self.embedding_norm(self.token_embedding(input_ids))
        # Both receptive-field variants are hoisted out of the layer loop: default
        # depth now performs two rather than eighteen O(T^2) mask constructions.
        additive_masks = self._attention_masks(attention_mask, batch, length, input_ids.device, segment_ids)
        # Compute RoPE once per forward; full layers ignore it, local layers reuse it.
        rope = self.rotary_embedding(positions, hidden.dtype)
        for layer in self.layers:
            # NoPE is selected by layer type rather than theta=0: applying RoPE with
            # theta zero would be undefined and identity RoPE would still encode pos.
            layer_rope = (
                None
                if self.config.use_nope_global and layer.attention.layer_type == "full_attention"
                else rope
            )
            layer_mask = additive_masks[layer.attention.layer_type]
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    lambda state, block=layer, block_rope=layer_rope, mask=layer_mask: block(state, block_rope, mask),
                    hidden, use_reentrant=False,
                )
            else:
                hidden = layer(hidden, layer_rope, layer_mask)
        # Project in model dtype, then use FP32 for scaling/cap/loss stability. The
        # width-derived multiplier keeps initial logit variance comparable by scale.
        logits = self.lm_head(self.final_norm(hidden)).float() * self.config.logit_multiplier
        cap = self.config.final_logit_softcap
        if cap > 0:
            # cap*tanh(x/cap) is linear near zero and smoothly saturates at +/-cap,
            # avoiding a hard clip's zero/discontinuous gradient boundary.
            logits = torch.tanh(logits / cap) * cap
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            # Standard next-token teacher forcing: prediction t targets token t+1;
            # reshape flattens batch/time and PyTorch CE honors label -100 masking.
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return CausalLMOutput(logits=logits, loss=loss)
