from pathlib import Path

import json
import pytest

from src.tokenization import load_tokenizer
from src.training.proxy_experiment import _dataset_sha256, _tokenize, _tokenize_row, run


TOKENIZER = Path("artifacts/tokenizers/v2/tiktoken-tr-bpe-12k.json")


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


def test_tokenize_can_select_a_split_from_shared_shards(tmp_path: Path):
    tokenizer = load_tokenizer(TOKENIZER)
    dataset = tmp_path / "shared.jsonl"
    train_text = "train-only text"
    validation_text = "validation-only text"
    dataset.write_text(
        "\n".join(
            (
                json.dumps({"split": "train", "text": train_text}),
                json.dumps({"split": "validation", "text": validation_text}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    train = _tokenize(dataset, tokenizer, len(tokenizer.encode(train_text)), split="train")
    validation = _tokenize(
        dataset, tokenizer, len(tokenizer.encode(validation_text)), split="validation"
    )
    assert train[0].input_ids == tuple(tokenizer.encode(train_text))
    assert validation[0].input_ids == tuple(tokenizer.encode(validation_text))


def test_tokenize_rejects_unknown_chat_role(tmp_path: Path):
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(
        json.dumps({"messages": [{"role": "alien", "content": "hello"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported message role"):
        _tokenize(dataset, load_tokenizer(TOKENIZER), 10)


def test_proxy_run_resumes_remaining_updates(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    text = "training tokens " * 200
    train.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
    validation.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
    config = {
        "format_version": 1,
        "experiment_id": "resume-test",
        "seed": 2026,
        "device": "cpu",
        "precision": "fp32",
        "tokenizer": str(TOKENIZER),
        "train_data": str(train),
        "validation_data": str(validation),
        "train_token_budget": 64,
        "validation_token_budget": 64,
        "sequence_length": 16,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer_updates": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "gradient_checkpointing": False,
        "max_grad_norm": 1.0,
        "total_steps": 2,
        "output_dir": str(tmp_path / "run"),
        "models": {
            "dt": {
                "vocab_size": 12000,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "max_position_embeddings": 16,
            }
        },
    }
    initial_config = tmp_path / "initial.json"
    initial_config.write_text(json.dumps(config), encoding="utf-8")
    run(initial_config, tmp_path / "initial-report.json")

    config["optimizer_updates"] = 2
    config["resume_from"] = {"dt": str(tmp_path / "run" / "dt" / "final.pt")}
    resumed_config = tmp_path / "resumed.json"
    resumed_config.write_text(json.dumps(config), encoding="utf-8")
    report = run(resumed_config, tmp_path / "resumed-report.json")
    result = report["results"][0]
    assert result["optimizer_updates"] == 2
    assert result["resume"]["starting_update"] == 1
    assert result["resume"]["completed_without_additional_training"] is False
