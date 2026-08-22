import pytest
import torch

from src.inference.cli import _checkpoint_fields, load_model, resolve_device, resolve_precision
from src.models import create_config, create_model
from src.tokenization import load_tokenizer, tokenizer_fingerprint


def test_checkpoint_envelopes_are_normalized():
    weights = {"weight": torch.ones(1)}
    standalone = {
        "model_type": "dt", "config": {"vocab_size": 10},
        "model_state": weights, "step": 7, "metadata": {"variant": "control"},
    }
    assert _checkpoint_fields(standalone)[3:] == (7, "standalone", {"variant": "control"})
    trainer = {
        "model_type": "bit", "model_config": {"vocab_size": 10},
        "model_state": weights, "trainer_state": {"update_step": 11},
    }
    assert _checkpoint_fields(trainer)[3:5] == (11, "trainer")
    model_only = {
        **trainer, "checkpoint_kind": "model_only_scaling",
        "metadata": {"variant": "bit-channel"},
    }
    assert _checkpoint_fields(model_only)[3:] == (
        11, "model_only_scaling", {"variant": "bit-channel"}
    )


def test_checkpoint_validation_rejects_missing_identity():
    with pytest.raises(ValueError, match="model_type"):
        _checkpoint_fields({"model_state": {}})


def test_device_and_precision_resolution_are_safe():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_precision("auto", torch.device("cpu")) == "fp32"
    with pytest.raises(ValueError, match="requires CUDA"):
        resolve_precision("fp16", torch.device("cpu"))


def test_checkpoint_rejects_same_size_wrong_tokenizer(tmp_path):
    expected = load_tokenizer("artifacts/tokenizers/v2/tiktoken-style-tr-bpe-12k.json")
    config = create_config(
        "dt", vocab_size=12000, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=16,
    )
    model = create_model(config)
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "model_type": "dt", "model_config": config.to_dict(),
        "model_state": model.state_dict(),
        "metadata": {"tokenizer_fingerprint": tokenizer_fingerprint(expected)},
    }, checkpoint)
    with pytest.raises(ValueError, match="fingerprint"):
        load_model(checkpoint, "artifacts/tokenizers/v2/byte-bpe-12k.json", torch.device("cpu"))
