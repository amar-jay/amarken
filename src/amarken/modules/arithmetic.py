"""Arithmetic SFT/RL hooks: masked loss, exact-match rewards and held-out metrics."""

import re

import torch
from torchmetrics import Metric

from amarken.constants import ARITHMETIC_MAX_NEW_TOKENS, SFT_SYSTEM_PROMPT
from amarken.modules.causal import build_model, fsdp_modules, load_model
from amarken.modules.causal import loss_fn as causal_loss
from amarken.modules.causal import predict as generate_text

__all__ = [
    "build_metrics",
    "build_model",
    "fsdp_modules",
    "load_model",
    "loss_fn",
    "predict",
    "prepare_prompts",
    "reward_fn",
    "validation_step",
]

SUPPORTS_RL = True


def prepare_prompts(tokenizer, prompts):
    return [
        tokenizer.apply_chat_template(
            (
                [{"role": "system", "content": SFT_SYSTEM_PROMPT}]
                if SFT_SYSTEM_PROMPT
                else []
            )
            + [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]


def parse_answer(text):
    text = text.strip()
    # Reject explanations, repeated answers and other reward shortcuts.
    return int(text) if re.fullmatch(r"[+-]?[0-9]{1,6}", text) else None


def _answer_reward(text, expected):
    return float(parse_answer(text) == int(expected))


def reward_fn(completions, answer, **kwargs):
    """TRL reward callback; answer is an arithmetic dataset column.

    TRL also supplies prompts, completion_ids, trainer_state and extra columns.
    This task only needs completion text and its reference integer.
    """
    return [
        _answer_reward(text, expected)
        for text, expected in zip(completions, answer, strict=True)
    ]


def predict(model, inputs):
    options = inputs if isinstance(inputs, dict) else {"texts": inputs}
    prompts = options["texts"]
    prompts = [prompts] if isinstance(prompts, str) else prompts
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Provide nonempty arithmetic prompts in texts")
    result = generate_text(
        model,
        {
            "texts": prepare_prompts(model.tokenizer, prompts),
            "max_new_tokens": options.get("max_new_tokens", ARITHMETIC_MAX_NEW_TOKENS),
        },
    )
    return {**result, "answers": [parse_answer(text) for text in result["texts"]]}


class ArithmeticMetrics(Metric):
    def __init__(self):
        super().__init__()
        for name in ("correct", "formatted", "characters", "samples"):
            self.add_state(
                name, default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
            )

    def update(self, texts, answers):
        self.correct += sum(
            int(_answer_reward(text, answer))
            for text, answer in zip(texts, answers, strict=True)
        )
        self.formatted += sum(parse_answer(text) is not None for text in texts)
        self.characters += sum(len(text) for text in texts)
        self.samples += len(texts)

    def compute(self):
        return {
            "accuracy": self.correct.float() / self.samples,
            "format_rate": self.formatted.float() / self.samples,
            "response_characters": self.characters.float() / self.samples,
        }


def build_metrics():
    return ArithmeticMetrics()


def validation_step(model, batch, device, metrics):
    prompts, answers = batch["prompts"], batch["answers"]
    result = predict(model, prompts)
    metrics.update(result["texts"], answers.tolist())
    return {
        "reward": sum(
            _answer_reward(text, answer)
            for text, answer in zip(result["texts"], answers, strict=True)
        )
        / len(prompts)
    }


def loss_fn(model, batch, device):
    """Response-only arithmetic SFT; GRPO uses the reward hook instead."""
    if "inputs" not in batch or "labels" not in batch:
        raise TypeError("Arithmetic SFT requires batches built with model.tokenize")
    return causal_loss(model, (batch["inputs"], batch["labels"]), device)
