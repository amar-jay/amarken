import json
from pathlib import Path

from src.models import create_config, create_model


def test_proxy_10m_arms_are_matched_controls():
    config = json.loads(Path("configs/proxy_10m.json").read_text())
    models = {name: create_model(create_config(name, **values)) for name, values in config["models"].items()}
    assert set(models) == {"dt", "glimmer", "bit"}
    assert {model.config.vocab_size for model in models.values()} == {12_000}
    assert {model.config.max_position_embeddings for model in models.values()} == {config["sequence_length"]}
    parameters = {name: model.parameter_count() for name, model in models.items()}
    assert all(9_000_000 <= count <= 11_000_000 for count in parameters.values())
    assert max(parameters.values()) / min(parameters.values()) < 1.01
    flops = {name: model.stats(config["sequence_length"]).flops_per_token for name, model in models.items()}
    assert flops["dt"] == flops["bit"]
    assert max(flops.values()) / min(flops.values()) < 1.02
    assert config["gradient_accumulation_steps"] * config["batch_size"] * config["sequence_length"] * config["optimizer_updates"] == 4_096
