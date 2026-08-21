"""Shared runtime contract for every Amarken causal language model.

The interface deliberately owns experiment-facing behavior (outputs, generation,
accounting, checkpoints) while architectures own only their forward computation
and exact FLOP/KV hooks. This prevents evaluation code from special-casing models.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, ClassVar, Generic, Protocol, TypeVar

import torch
from torch import Tensor, nn


class ModelConfig(Protocol):
    """Minimum immutable config surface consumed by shared infrastructure."""

    model_type: ClassVar[str]
    vocab_size: int
    max_position_embeddings: int

    def to_dict(self) -> dict[str, Any]: ...


ConfigT = TypeVar("ConfigT", bound=ModelConfig)


@dataclass
class CausalLMOutput:
    """Identical forward result for training and inference across architectures."""

    logits: Tensor
    loss: Tensor | None = None


@dataclass(frozen=True)
class ModelStats:
    """Comparable analytical resource accounting at a specified sequence length.

    FLOPs include learned matrix multiply-adds (two FLOPs per MAC), QK products,
    and attention-value products. Elementwise ops, normalization, RoPE, softmax,
    embedding lookup and quantizer bookkeeping are excluded and disclosed here so
    estimates remain deterministic rather than pretending to be hardware profiles.
    """

    model_type: str
    sequence_length: int
    total_parameters: int
    active_parameters: int
    ternary_parameters: int
    floating_parameters: int
    forward_flops: int
    flops_per_token: float
    artifact_bytes: int
    training_parameter_bytes: int
    kv_cache_bytes: int


@dataclass(frozen=True)
class CheckpointInfo:
    """Non-weight state returned after restoring a training checkpoint."""

    step: int
    metadata: dict[str, Any]


class AmarkenCausalLM(nn.Module, ABC, Generic[ConfigT]):
    """Common model API implemented once for the full student tournament."""

    config_type: ClassVar[type]
    config: ConfigT

    def parameter_count(self, trainable_only: bool = False) -> int:
        # parameters() deduplicates tied tensors, which is the project's counting rule.
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

    @abstractmethod
    def _resource_accounting(self, sequence_length: int, element_bytes: int) -> tuple[int, int, int, int]:
        """Return (forward FLOPs, KV bytes, artifact bytes, ternary parameters)."""

    def stats(self, sequence_length: int, element_bytes: int = 2) -> ModelStats:
        """Return architecture-aware counts in a single comparison schema."""
        if not 1 <= sequence_length <= self.config.max_position_embeddings:
            raise ValueError("sequence_length must be within the model context limit")
        if element_bytes < 1:
            raise ValueError("element_bytes must be positive")
        total = self.parameter_count()
        forward_flops, kv_bytes, artifact_bytes, ternary = self._resource_accounting(
            sequence_length, element_bytes
        )
        return ModelStats(
            model_type=self.config.model_type,
            sequence_length=sequence_length,
            total_parameters=total,
            active_parameters=total,  # Both current models are dense; no conditional experts.
            ternary_parameters=ternary,
            floating_parameters=total - ternary,
            forward_flops=forward_flops,
            flops_per_token=forward_flops / sequence_length,
            artifact_bytes=artifact_bytes,
            # Training master count excludes gradients and optimizer states by design.
            training_parameter_bytes=sum(p.numel() * p.element_size() for p in self.parameters()),
            kv_cache_bytes=kv_bytes,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Shared no-cache autoregressive generation for correctness/evaluation.

        This intentionally favors identical semantics over speed. A future common
        KV-cache API can optimize both models without changing callers.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [batch, nonempty sequence]")
        if max_new_tokens < 0 or temperature < 0:
            raise ValueError("max_new_tokens and temperature must be nonnegative")
        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k must be within the vocabulary")
        tokens = input_ids
        # Preserve None for unpadded batches so architectures can select native
        # causal/Flash-compatible attention instead of receiving a dense mask.
        mask = None if attention_mask is None else attention_mask.bool()
        if mask is not None and mask.shape != tokens.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # No-cache fallback retains only the configured context. Cropping is
                # deterministic and keeps mask/token alignment for long generations.
                context_tokens = tokens[:, -self.config.max_position_embeddings :]
                context_mask = None if mask is None else mask[:, -self.config.max_position_embeddings :]
                logits = self(context_tokens, attention_mask=context_mask).logits[:, -1]
                if temperature == 0.0:
                    next_token = logits.argmax(dim=-1)
                else:
                    sample_logits = logits / temperature
                    if top_k is not None:
                        threshold = torch.topk(sample_logits, top_k, dim=-1).values[:, -1:]
                        sample_logits = sample_logits.masked_fill(sample_logits < threshold, float("-inf"))
                    probabilities = torch.softmax(sample_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
                if eos_token_id is not None:
                    # Keep finished rows stable while unfinished batch members continue.
                    next_token = torch.where(finished, eos_token_id, next_token)
                    finished |= next_token.eq(eos_token_id)
                tokens = torch.cat((tokens, next_token[:, None]), dim=1)
                if mask is not None:
                    mask = torch.cat((mask, torch.ones_like(next_token[:, None], dtype=torch.bool)), dim=1)
                if eos_token_id is not None and finished.all():
                    break
        finally:
            self.train(was_training)
        return tokens

    def _checkpoint_payload(
        self,
        optimizer: torch.optim.Optimizer | None,
        step: int,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": 1,
            "model_type": self.config.model_type,
            # asdict stores constructor fields only; computed display fields must not
            # leak into config reconstruction (notably Glimmer layer_types).
            "config": asdict(self.config),
            "model_state": self.state_dict(),
            "step": step,
            "metadata": {} if metadata is None else metadata,
            "cpu_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        return payload

    def save_checkpoint(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        step: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Atomically save model plus optional resumable training state."""
        if step < 0:
            raise ValueError("step must be nonnegative")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(self._checkpoint_payload(optimizer, step, metadata), temporary)
        os.replace(temporary, destination)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> tuple["AmarkenCausalLM", CheckpointInfo]:
        """Construct a model and restore weights; reject cross-architecture loads."""
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        if payload.get("model_type") != cls.config_type.model_type:
            raise ValueError(f"checkpoint is {payload.get('model_type')!r}, not {cls.config_type.model_type!r}")
        model = cls(cls.config_type(**payload["config"]))
        model.load_state_dict(payload["model_state"], strict=True)
        return model, CheckpointInfo(int(payload.get("step", 0)), dict(payload.get("metadata", {})))

    def restore_training_state(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> CheckpointInfo:
        """Restore an existing model and optional optimizer/RNG for exact resume."""
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if payload.get("model_type") != self.config.model_type or payload.get("config") != asdict(self.config):
            raise ValueError("checkpoint architecture/config does not match this model")
        self.load_state_dict(payload["model_state"], strict=True)
        if optimizer is not None:
            if "optimizer_state" not in payload:
                raise ValueError("checkpoint does not contain optimizer state")
            optimizer.load_state_dict(payload["optimizer_state"])
        if restore_rng:
            torch.set_rng_state(payload["cpu_rng_state"].cpu())
            if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
                torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        return CheckpointInfo(int(payload.get("step", 0)), dict(payload.get("metadata", {})))
