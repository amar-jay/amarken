import json
from pathlib import Path

from src.models import create_config, create_model
from src.training.scaling_experiment import _rotated_order


def test_scaling_matrix_matches_parameters_flops_seeds_and_tokens():
    config = json.loads(Path("configs/proxy_scaling.json").read_text())
    assert len(config["seeds"]) >= 3 and len(set(config["seeds"])) == len(config["seeds"])
    assert set(config["scales"]) == {"25m", "60m"}
    expected = config["optimizer_updates"] * config["gradient_accumulation_steps"] * config["batch_size"] * config["sequence_length"]
    assert expected == 4_096
    for scale, candidates in config["scales"].items():
        models = {name: create_model(create_config(name, **values)) for name, values in candidates.items()}
        assert set(models) == {"dt", "glimmer", "bit"}
        parameters = [model.parameter_count() for model in models.values()]
        flops = [model.stats(config["sequence_length"]).flops_per_token for model in models.values()]
        target = 25_000_000 if scale == "25m" else 60_000_000
        assert all(abs(count - target) / target < 0.03 for count in parameters)
        assert max(parameters) / min(parameters) < 1.02
        assert max(flops) / min(flops) < 1.02


def test_architecture_order_rotates_without_changing_membership():
    orders = [_rotated_order(index) for index in range(3)]
    assert all(set(order) == {"dt", "glimmer", "bit"} for order in orders)
    assert [order[0] for order in orders] == ["dt", "glimmer", "bit"]
