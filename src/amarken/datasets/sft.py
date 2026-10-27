"""Single-turn instruction/response SFT using the selected model's chat template.

text_column maps the instruction, label_column the response, and optional
input_column adds user context. Labels supervise only the response and EOS.
"""

import hashlib
import re

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from amarken.constants import (
    CAUSAL_MAX_LENGTH,
    DATA_CACHE,
    DATASET_REVISIONS,
    SFT_SYSTEM_PROMPT,
)
from amarken.utils.distributed import train_sampler


def _prompt(row, config):
    instruction = row.get(config.text_column)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"SFT needs a nonempty instruction in {config.text_column!r}")
    context = row.get(config.input_column) if config.input_column else ""
    if not isinstance(context, str):
        raise TypeError(f"SFT context in {config.input_column!r} must be a string")
    return instruction.strip() + ("\n\n" + context.strip() if context.strip() else "")


def _partition(prompt):
    # Prompt groups stay together even if multiple responses exist for one prompt.
    return int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8], "big") / 2**64


def _build_dataset(config, *, seed, test=False):
    splits = list(dict.fromkeys([config.train_split, config.validation_split]))
    sources = load_dataset(
        config.id,
        name=config.config,
        split=splits,
        revision=DATASET_REVISIONS.get(config.id),
        cache_dir=DATA_CACHE,
    )
    sources = dict(zip(splits, sources, strict=True))
    train = sources[config.train_split]
    heldout = sources[config.validation_split]
    required = {config.text_column, config.label_column}
    if config.input_column:
        required.add(config.input_column)
    for source in sources.values():
        if not required.issubset(source.column_names):
            raise ValueError(f"SFT dataset requires columns {sorted(required)}")
    shared = config.train_split == config.validation_split
    heldout_prompts = None if shared else {_prompt(row, config) for row in heldout}
    train_indices = []
    if not test:
        for index, row in enumerate(train):
            prompt = _prompt(row, config)
            if (
                (_partition(prompt) < 0.8)
                if shared
                else (prompt not in heldout_prompts)
            ):
                train_indices.append(index)
    eval_indices = []
    boundary = 1 - (0.2 if shared else 1) * config.test_fraction
    for index, row in enumerate(heldout):
        value = _partition(_prompt(row, config))
        if shared and value < 0.8:
            continue
        if (value >= boundary) == test:
            eval_indices.append(index)

    def sample(source, indices, size):
        if not indices:
            raise ValueError("SFT partition is empty; use a larger source split")
        selected = source.select(indices).shuffle(seed=seed)
        return selected.select(range(min(size, len(selected))))

    evaluation = sample(
        heldout, eval_indices, config.test_size if test else config.val_size
    )
    if test:
        return evaluation
    return sample(train, train_indices, config.train_size), evaluation


def build_dataset(config, *, seed=42):
    return _build_dataset(config, seed=seed)


def build_test_dataset(config, *, seed=42):
    return _build_dataset(config, seed=seed, test=True)


def encode_sample(tokenizer, row, config):
    """Verify the response span before tokenizing the complete conversation.

    Fast-tokenizer offsets avoid assuming separately tokenized prompt IDs are a
    prefix of full IDs. Tokens crossing the boundary are ignored conservatively.
    Native assistant masks, when available, are intersected with the verified
    response span. EOS is supervised even if the native mask excludes it.
    """
    response = row.get(config.label_column)
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"SFT needs a nonempty response in {config.label_column!r}")
    response = response.strip()
    messages = []
    if SFT_SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SFT_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": _prompt(row, config)})
    prefix = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    messages.append({"role": "assistant", "content": response})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    start, end = len(prefix), len(prefix) + len(response)
    eos = tokenizer.eos_token
    if (
        not eos
        or not rendered.startswith(prefix + response + eos)
        or rendered[end + len(eos) :].strip()
    ):
        raise ValueError(
            "SFT chat template must preserve the prompt prefix and end the response with EOS"
        )
    if not tokenizer.is_fast:
        raise ValueError("SFT response masking requires a fast tokenizer with offsets")
    native = bool(re.search(r"{%[-+]?\s*generation\b", tokenizer.get_chat_template()))
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_assistant_tokens_mask=native,
        tokenizer_kwargs={"return_offsets_mapping": True},
    )
    ids = encoded["input_ids"]
    if len(ids) > CAUSAL_MAX_LENGTH:
        raise ValueError(
            f"SFT sample has {len(ids)} tokens; exceeds CAUSAL_MAX_LENGTH={CAUSAL_MAX_LENGTH}. "
            "Shorten the sample or increase the constant; truncation is disabled."
        )
    labels = []
    content_targets = 0
    for index, (token, (left, right)) in enumerate(
        zip(ids, encoded["offset_mapping"], strict=True)
    ):
        content = start <= left < right <= end
        if native:
            content = content and bool(encoded["assistant_masks"][index])
        ending = (
            end <= left < right <= end + len(eos) and token == tokenizer.eos_token_id
        )
        labels.append(token if content or ending else -100)
        content_targets += content and index > 0
    if not content_targets or tokenizer.eos_token_id not in labels:
        raise ValueError(
            "SFT sample must retain response tokens and the ending EOS target"
        )
    return ids, labels


def build_dataloaders(config, train, validation, *, batch_size, tokenize=None):
    # The common interface passes model.tokenize, a bound method. Formatting
    # stays here; only its owner's tokenizer is needed, not a model forward.
    tokenizer = getattr(getattr(tokenize, "__self__", None), "tokenizer", None)
    if (
        tokenizer is None
        or not tokenizer.chat_template
        or tokenizer.pad_token_id is None
    ):
        raise TypeError(
            "SFT requires model.tokenize with a tokenizer, chat template and PAD token"
        )

    def collate(samples):
        encoded = [encode_sample(tokenizer, row, config) for row in samples]
        width = max(len(ids) for ids, _ in encoded)
        ids = torch.full(
            (len(samples), width), tokenizer.pad_token_id, dtype=torch.long
        )
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for index, (tokens, targets) in enumerate(encoded):
            ids[index, : len(tokens)] = torch.tensor(tokens)
            mask[index, : len(tokens)] = 1
            labels[index, : len(tokens)] = torch.tensor(targets)
        return {"input_ids": ids, "attention_mask": mask}, labels

    sampler = train_sampler(train)
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=collate,
            num_workers=0,
        ),
        DataLoader(
            validation, batch_size=batch_size, collate_fn=collate, num_workers=0
        ),
    )
