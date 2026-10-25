"""Pinned Qwen feature backbone; no generation or decision head."""
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from amarken.constants import BASES, REVISIONS


@dataclass(frozen=True)
class BaseConfig:
    model_id: str = BASES[0]
    revision: str | None = None
    max_length: int = 2048
    device: str = "cpu"
    dtype: str = "float32"

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
    def load(cls, config: BaseConfig | None = None, *, local_files_only=False):
        config = config or BaseConfig()
        common = dict(revision=config.revision, trust_remote_code=False,
                      local_files_only=local_files_only)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, **common)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModel.from_pretrained(
            config.model_id, dtype=getattr(torch, config.dtype), **common
        ).to(config.device)
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        if config.max_length > model.config.max_position_embeddings:
            raise ValueError("max_length exceeds the model context window")
        return cls(model, tokenizer, config).eval()

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def tokenize(self, texts: Sequence[str]) -> dict[str, Tensor]:
        if isinstance(texts, str) or not texts or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ValueError("Provide a nonempty batch of nonempty strings")
        batch = self.tokenizer(list(texts), padding=True, truncation=False,
                               return_tensors="pt", return_attention_mask=True)
        if batch["input_ids"].shape[1] > self.settings.max_length:
            raise ValueError("Input exceeds max_length; truncation is disabled")
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in batch.items()}

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        # No inference/no_grad decorator: callers can fine-tune these features.
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("Expected matching [batch, tokens] inputs and mask")
        if input_ids.shape[1] > self.settings.max_length:
            raise ValueError("Input exceeds max_length")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)) or not torch.all(attention_mask.any(dim=1)):
            raise ValueError("Invalid or empty attention mask")
        result = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False, return_dict=True)
        return pool_last_token(result.last_hidden_state, attention_mask)
