from pathlib import Path

import pytest

from src.tokenization.tokenizer import load_tokenizer


TOKENIZER = Path("artifacts/tokenizers/v3/tiktoken-tr-bpe-12k.json")


def test_chat_boundary_ids_use_role_tokens_and_end_token():
    tokenizer = load_tokenizer(TOKENIZER)

    assert tokenizer.system_id() == tokenizer.token_to_id("<|system|>")
    assert tokenizer.user_id() == tokenizer.token_to_id("<|user|>")
    assert tokenizer.assistant_id() == tokenizer.token_to_id("<|assistant|>")
    assert tokenizer.end_id() == tokenizer.token_to_id("<|end|>")
    assert tokenizer.eos_id() == tokenizer.end_id()
    assert tokenizer.start_id("assistant") == tokenizer.assistant_id()


def test_chat_start_requires_an_explicit_role():
    tokenizer = load_tokenizer(TOKENIZER)
    with pytest.raises(ValueError, match="role must"):
        tokenizer.start_id("bos")
