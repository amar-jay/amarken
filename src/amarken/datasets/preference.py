"""Preference pairs for DPO, with prompt-group splits and no token truncation."""

import hashlib
import json

from datasets import Dataset, load_dataset

from amarken.constants import DATA_CACHE, DATASET_REVISIONS, SFT_SYSTEM_PROMPT


def _pair(row):
    chosen, rejected = row.get("chosen"), row.get("rejected")
    if isinstance(chosen, list) and isinstance(rejected, list):
        if (
            len(chosen) < 2
            or len(rejected) < 2
            or any(not isinstance(message, dict) for message in chosen + rejected)
            or chosen[:-1] != rejected[:-1]
            or chosen[-1].get("role") != "assistant"
            or rejected[-1].get("role") != "assistant"
        ):
            raise ValueError(
                "Preference conversations need a shared prompt and final assistant responses"
            )
        messages = chosen[:-1]
        chosen, rejected = chosen[-1].get("content"), rejected[-1].get("content")
    else:
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Preference data needs a nonempty prompt")
        messages = (
            [{"role": "system", "content": SFT_SYSTEM_PROMPT}]
            if SFT_SYSTEM_PROMPT
            else []
        )
        messages += [{"role": "user", "content": prompt.strip()}]
    if any(
        not isinstance(text, str) or not text.strip() for text in (chosen, rejected)
    ):
        raise ValueError("Preference data needs nonempty chosen and rejected responses")
    if chosen.strip() == rejected.strip():
        raise ValueError("Chosen and rejected responses must differ")
    if any(
        not isinstance(m, dict)
        or not isinstance(m.get("content"), str)
        or m.get("role") not in {"system", "user", "assistant"}
        for m in messages
    ):
        raise ValueError(
            "Preference prompt messages must contain text and supported roles"
        )
    return messages, chosen.strip(), rejected.strip()


def _build(config, *, seed, tokenize=None, max_length=None, split=None):
    if split not in {None, "val", "test"}:
        raise ValueError("Preference split must be val or test")
    tokenizer = getattr(getattr(tokenize, "__self__", None), "tokenizer", None)
    if tokenize is not None and (
        tokenizer is None or not tokenizer.chat_template or tokenizer.eos_token is None
    ):
        raise TypeError(
            "Preference preparation requires model.tokenize with a chat template and EOS"
        )
    names = list(dict.fromkeys([config.train_split, config.validation_split]))
    sources = load_dataset(
        config.id,
        name=config.config,
        split=names,
        revision=DATASET_REVISIONS.get(config.id),
        cache_dir=DATA_CACHE,
    )
    sources = dict(zip(names, sources, strict=True))

    def sample(partition, size):
        source = sources[
            config.train_split if partition == 0 else config.validation_split
        ]
        if not {"chosen", "rejected"}.issubset(source.column_names):
            raise ValueError("Preference data requires chosen and rejected columns")
        result, seen = [], set()
        for row in source.shuffle(seed=seed + partition):
            try:
                messages, chosen, rejected = _pair(row)
            except ValueError:
                # Published preference sources can contain empty, identical or
                # inconsistent pairs; they provide no valid preference target.
                continue
            key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            fraction = (
                int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
            )
            group = (
                0
                if fraction < 0.8
                else (1 if fraction < 1 - 0.2 * config.test_fraction else 2)
            )
            if group != partition or key in seen:
                continue
            prompt = messages
            if tokenizer is not None:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if not chosen.endswith(tokenizer.eos_token):
                    chosen += tokenizer.eos_token
                if not rejected.endswith(tokenizer.eos_token):
                    rejected += tokenizer.eos_token
                if any(
                    len(tokenizer.encode(prompt + response, add_special_tokens=False))
                    > max_length
                    for response in (chosen, rejected)
                ):
                    continue
            seen.add(key)
            result.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            if len(result) == size:
                break
        if len(result) < size:
            raise ValueError(
                f"Preference partition {partition} has only {len(result)} eligible pairs; requested {size}. Reduce sizes or increase max_length."
            )
        return Dataset.from_list(result)

    if split is not None:
        return sample(
            1 if split == "val" else 2,
            config.val_size if split == "val" else config.test_size,
        )
    return sample(0, config.train_size), sample(1, config.val_size)


def build_dataset(config, *, seed=42):
    return _build(config, seed=seed)


def build_test_dataset(config, *, seed=42):
    return _build(config, seed=seed, split="test")


def build_preference_dataset(config, *, tokenize, max_length, seed=42, split=None):
    """Return TRL-ready prompt/chosen/rejected records, using pretrained EOS."""
    return _build(
        config, seed=seed, tokenize=tokenize, max_length=max_length, split=split
    )


def build_dataloaders(*args, **kwargs):
    raise NotImplementedError(
        "Preference pairs use the DPO adapter's native dataloaders and evaluation"
    )
