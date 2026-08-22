import pytest
import torch

from src.inference.cli import _checkpoint_fields, resolve_device, resolve_precision


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
