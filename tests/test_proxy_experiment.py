from pathlib import Path

import json
import pytest

from src.tokenization import load_tokenizer
from src.training.proxy_experiment import _dataset_sha256, _tokenize, _tokenize_row


TOKENIZER = Path("artifacts/tokenizers/v2/tiktoken-style-tr-bpe-12k.json")


def test_tokenize_supports_flat_text_and_assistant_masked_chat_shards(tmp_path: Path):
    tokenizer = load_tokenizer(TOKENIZER)
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "000.jsonl").write_text(
        json.dumps({"text": "plain pretraining text"}) + "\n", encoding="utf-8"
    )
    (shards / "001.jsonl").write_text(
        json.dumps({
            "messages": [
                {"role": "user", "content": "Translate hello"},
                {"role": "assistant", "content": "Merhaba"},
            ]
        }) + "\n",
        encoding="utf-8",
    )
    flat_tokens = len(tokenizer.encode("plain pretraining text"))
    chat = _tokenize_row({
        "messages": [
            {"role": "user", "content": "Translate hello"},
            {"role": "assistant", "content": "Merhaba"},
        ]
    }, tokenizer)
    assert chat is not None
    examples = _tokenize(shards, tokenizer, flat_tokens + len(chat.input_ids))
    assert all(examples[0].assistant_mask)
    assert not all(examples[1].assistant_mask)
    assert any(examples[1].assistant_mask)
    assert sum(len(example.input_ids) for example in examples) == flat_tokens + len(chat.input_ids)


def test_dataset_hash_binds_shard_names_and_contents(tmp_path: Path):
    dataset = tmp_path / "shards"
    dataset.mkdir()
    shard = dataset / "a.jsonl"
    shard.write_text('{"text":"one"}\n', encoding="utf-8")
    first = _dataset_sha256(dataset)
    shard.rename(dataset / "b.jsonl")
    assert _dataset_sha256(dataset) != first


def test_tokenize_rejects_unknown_chat_role(tmp_path: Path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(
        json.dumps({"messages": [{"role": "alien", "content": "hello"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported message role"):
        _tokenize(dataset, load_tokenizer(TOKENIZER), 10)
