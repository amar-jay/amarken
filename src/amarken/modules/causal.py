"""Causal language modeling with full fine-tuning or PEFT LoRA.

Batches are (tokenized_inputs, labels); labels use -100 for ignored positions.
This is ordinary next-token training. Instruction/response masking belongs in
an SFT dataset collator, which can use this same module without changing Trainer.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import nn
from torchmetrics import Metric
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from amarken.config import FineTuningConfig
from amarken.constants import (
    BASE_LOCAL_FILES_ONLY,
    CAUSAL_MAX_LENGTH,
    CAUSAL_MAX_NEW_TOKENS,
    HUB_CACHE,
    REVISIONS,
)
from amarken.utils.helpers import load_checkpoint_weights


class CausalLM(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def tokenize(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        if (
            isinstance(texts, str)
            or not texts
            or any(not isinstance(text, str) or not text.strip() for text in texts)
        ):
            raise ValueError("Provide a nonempty batch of nonempty strings")
        encoded = [
            self.tokenizer.encode(text, add_special_tokens=False)
            + [self.tokenizer.eos_token_id]
            for text in texts
        ]
        if any(len(row) < 2 or len(row) > CAUSAL_MAX_LENGTH for row in encoded):
            raise ValueError(
                f"Training text needs 2..{CAUSAL_MAX_LENGTH} tokens including EOS"
            )
        # Right padding during training; mask by position because PAD may equal EOS.
        width = max(map(len, encoded))
        ids = torch.full(
            (len(encoded), width), self.tokenizer.pad_token_id, dtype=torch.long
        )
        mask = torch.zeros_like(ids)
        for index, row in enumerate(encoded):
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def forward(self, inputs: Mapping[str, torch.Tensor]):
        return self.model(**inputs, use_cache=False, return_dict=True).logits


def _build(*, model_id, device, precision, fine_tuning, pretrained):
    if model_id not in REVISIONS:
        raise ValueError("Set model_id and add its pinned revision to REVISIONS")
    settings = (
        FineTuningConfig(**fine_tuning)
        if isinstance(fine_tuning, dict)
        else fine_tuning
    )
    settings = settings or FineTuningConfig()
    common = {
        "revision": REVISIONS[model_id],
        "cache_dir": HUB_CACHE,
        "trust_remote_code": False,
        "local_files_only": BASE_LOCAL_FILES_ONLY,
    }
    architecture = AutoConfig.from_pretrained(model_id, **common)
    if architecture.model_type not in {"qwen2", "qwen3"}:
        raise ValueError(
            "The causal module currently supports Qwen2/Qwen2.5 and Qwen3 text models"
        )
    if CAUSAL_MAX_LENGTH > architecture.max_position_embeddings:
        raise ValueError("CAUSAL_MAX_LENGTH exceeds the model context window")
    tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
    if tokenizer.eos_token_id is None:
        raise ValueError("Causal training requires an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, precision)
    if pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, config=architecture, dtype=dtype, **common
        )
    else:
        model = AutoModelForCausalLM.from_config(architecture, dtype=dtype)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    if settings.mode == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=settings.r,
                lora_alpha=settings.alpha,
                lora_dropout=settings.dropout,
                target_modules=list(settings.target_modules),
                bias="none",
            ),
        )
    else:
        model.requires_grad_(True)
    return CausalLM(model, tokenizer).to(device)


def build_model(*, model_id=None, device="cpu", precision="float32", fine_tuning=None):
    return _build(
        model_id=model_id,
        device=device,
        precision=precision,
        fine_tuning=fine_tuning,
        pretrained=True,
    )


def load_model(
    checkpoint: str | Path,
    *,
    model_id=None,
    device="cpu",
    precision="float32",
    fine_tuning=None,
):
    # Full amarken states include frozen base weights as well as adapters.
    # Rebuild from config, avoiding a second pretrained-weight download.
    model = _build(
        model_id=model_id,
        device="cpu",
        precision=precision,
        fine_tuning=fine_tuning,
        pretrained=False,
    )
    model.load_state_dict(load_checkpoint_weights(checkpoint), strict=True)
    return model.to(device).eval()


def _token_losses(logits, labels):
    logits, targets = logits[:, :-1].float(), labels[:, 1:]
    valid = targets != -100
    if not valid.any(dim=1).all():
        raise ValueError("Each sample needs at least one unmasked next-token target")
    losses = nn.functional.cross_entropy(
        logits.transpose(1, 2), targets, ignore_index=-100, reduction="none"
    )
    # Sample means match the trainer's existing sample-weighted aggregation.
    loss = ((losses * valid).sum(1) / valid.sum(1)).mean()
    return loss, losses, valid, logits.argmax(-1), targets


def _forward_batch(model, batch, device):
    inputs, labels = batch
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return _token_losses(model(inputs), labels.to(device))


def loss_fn(model, batch, device):
    return _forward_batch(model, batch, device)[0]


class TokenMetrics(Metric):
    """Token-weighted NLL, perplexity and accuracy across the whole split."""

    def __init__(self):
        super().__init__()
        self.add_state(
            "nll", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "tokens", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, losses, valid, predictions, targets):
        self.nll += losses[valid].double().sum()
        self.correct += ((predictions == targets) & valid).sum()
        self.tokens += valid.sum()

    def compute(self):
        nll = self.nll / self.tokens
        return {
            "token_nll": nll,
            "perplexity": nll.exp(),
            "token_accuracy": self.correct / self.tokens,
        }


def build_metrics():
    return TokenMetrics()


def validation_step(model, batch, device, metrics):
    loss, losses, valid, predictions, targets = _forward_batch(model, batch, device)
    metrics.update(losses, valid, predictions, targets)
    return {"loss": loss.item()}


def predict(model: CausalLM, inputs: str | Sequence[str] | dict) -> dict:
    options = inputs if isinstance(inputs, dict) else {"texts": inputs}
    texts = options["texts"]
    texts = [texts] if isinstance(texts, str) else texts
    if not texts or any(
        not isinstance(text, str) or not text.strip() for text in texts
    ):
        raise ValueError("Provide nonempty prompt strings in texts")
    max_new_tokens = options.get("max_new_tokens", CAUSAL_MAX_NEW_TOKENS)
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    tokenizer = model.tokenizer
    previous_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        batch = tokenizer(
            list(texts), padding=True, return_tensors="pt", add_special_tokens=False
        )
    finally:
        tokenizer.padding_side = previous_side
    context = model.model.config.max_position_embeddings
    if batch["input_ids"].shape[1] + max_new_tokens > context:
        raise ValueError("Prompt plus generation exceeds the model context window")
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    model.eval()
    with torch.inference_mode():
        generated = model.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    continuations = generated[:, batch["input_ids"].shape[1] :]
    return {"texts": tokenizer.batch_decode(continuations, skip_special_tokens=True)}


def fsdp_modules(model):
    names = set(model.model._no_split_modules or [])
    blocks = [
        module for module in model.model.modules() if type(module).__name__ in names
    ]
    if not blocks:
        raise ValueError("Define FSDP block boundaries for this causal architecture")
    return list(reversed(blocks)) + [model]
