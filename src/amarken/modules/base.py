"""Pinned Qwen feature backbone; no generation or decision head."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from amarken.constants import (
    BASE_LOCAL_FILES_ONLY,
    BASE_MAX_LENGTH,
    BASE_MODEL_ID,
    HUB_CACHE,
    REVISIONS,
)
from amarken.utils import load_checkpoint_weights


@dataclass(frozen=True)
class BaseConfig:
    model_id: str = BASE_MODEL_ID
    revision: str | None = None
    max_length: int = BASE_MAX_LENGTH
    device: str = "cpu"
    dtype: str = "float32"
    cache_dir: str = HUB_CACHE

    def __post_init__(self):
        if self.model_id not in REVISIONS:
            raise ValueError(f"Supported models: {tuple(REVISIONS)}")
        if self.revision is None:
            object.__setattr__(self, "revision", REVISIONS[self.model_id])
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported dtype")


def pool_last_token(hidden: Tensor, mask: Tensor) -> Tensor:
    """Select the last real token under either left or right padding."""
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("Expected hidden [batch, tokens, width] and matching mask")
    if not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask.any(dim=1)):
        raise ValueError("Each row needs a binary mask with at least one real token")
    positions = torch.arange(mask.shape[1], device=hidden.device).expand_as(mask)
    last = positions.masked_fill(mask == 0, -1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class BaseModel(nn.Module):
    def __init__(self, model: nn.Module, tokenizer, config: BaseConfig):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.settings = config

    @classmethod
    def load(
        cls,
        config: BaseConfig | None = None,
        *,
        local_files_only=BASE_LOCAL_FILES_ONLY,
        pretrained=True,
    ):
        config = config or BaseConfig()
        common = {
            "revision": config.revision,
            "cache_dir": config.cache_dir,
            "trust_remote_code": False,
            "local_files_only": local_files_only,
        }
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, **common)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if pretrained:
            model = AutoModel.from_pretrained(
                config.model_id, dtype=getattr(torch, config.dtype), **common
            )
        else:
            architecture = AutoConfig.from_pretrained(config.model_id, **common)
            model = AutoModel.from_config(
                architecture,
                dtype=getattr(torch, config.dtype),
                trust_remote_code=False,
            )
        model = model.to(config.device)
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        text_config = getattr(model.config, "text_config", model.config)
        text_config.use_cache = False
        text_config.pad_token_id = tokenizer.pad_token_id
        if config.max_length > text_config.max_position_embeddings:
            raise ValueError("max_length exceeds the model context window")
        return cls(model, tokenizer, config).eval()

    @property
    def hidden_size(self) -> int:
        return getattr(self.model.config, "text_config", self.model.config).hidden_size

    def tokenize(self, texts: Sequence[str]) -> dict[str, Tensor]:
        if (
            isinstance(texts, str)
            or not texts
            or any(not isinstance(text, str) or not text.strip() for text in texts)
        ):
            raise ValueError("Provide a nonempty batch of nonempty strings")
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if batch["input_ids"].shape[1] > self.settings.max_length:
            raise ValueError("Input exceeds max_length; truncation is disabled")
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in batch.items()}

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Accept one tokenized input dictionary, matching model(inputs) calls."""
        if not isinstance(inputs, Mapping):
            raise TypeError(
                "BaseModel expects a tokenized dictionary with input_ids and attention_mask"
            )
        if "input_ids" not in inputs or "attention_mask" not in inputs:
            raise ValueError("Inputs must include input_ids and attention_mask")
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        if not isinstance(input_ids, Tensor) or not isinstance(attention_mask, Tensor):
            raise TypeError("input_ids and attention_mask must be tensors")
        # No inference/no_grad decorator: callers can fine-tune these features.
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Expected matching [batch, tokens] inputs and mask")
        if input_ids.shape[1] > self.settings.max_length:
            raise ValueError("Input exceeds max_length")
        if not torch.all(
            (attention_mask == 0) | (attention_mask == 1)
        ) or not torch.all(attention_mask.any(dim=1)):
            raise ValueError("Invalid or empty attention mask")
        result = self.model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )
        return pool_last_token(result.last_hidden_state, attention_mask)


def build_model(*, device: str = "cpu", precision: str = "float32") -> BaseModel:
    """Start from the pretrained backbone configured in constants.py."""
    return BaseModel.load(BaseConfig(device=device, dtype=precision))


def load_model(
    checkpoint: str | Path, *, device: str = "cpu", precision: str = "float32"
) -> BaseModel:
    """Restore trained weights using the pinned architecture and tokenizer.

    Only config/tokenizer files use the Hub/cache; pretrained weights are not
    loaded again. Keep constants consistent with the training architecture.
    """
    weights = load_checkpoint_weights(checkpoint)
    model = BaseModel.load(BaseConfig(dtype=precision), pretrained=False)
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


def predict(model: BaseModel, inputs: str | Sequence[str]) -> dict:
    """Extract features from text with an already-loaded backbone.

    This backbone has no task head: add your own output interpretation here.
    """
    texts = [inputs] if isinstance(inputs, str) else inputs
    batch = model.tokenize(texts)
    model.eval()
    with torch.inference_mode():
        features = model(batch)
    return {"features": features.float().cpu().tolist()}


def loss_fn(model, batch, device):
    """Task-loss hook with the same signature as the numeric implementation.

    BaseModel returns pooled features, not classification logits or a language
    modeling loss. Implement your objective here once the task/targets are
    defined. Tokenized inputs should be moved to device and passed as one
    dictionary: features = model(inputs).
    """
    raise NotImplementedError(
        "Define your task's loss in amarken/modules/base.py:loss_fn. "
        "BaseModel currently returns pooled features and has no task objective."
    )
