import pytest
import torch
from torch import nn

from src.models.bit import BitLinear
from src.models.factory import create_config, create_model
from src.training.optimizer import OptimizerConfig, create_optimizer, parameter_groups


def _model(model_type: str):
    options = dict(
        vocab_size=67,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=16,
    )
    if model_type == "glimmer":
        options["sliding_window"] = 8
    return create_model(create_config(model_type, **options))


@pytest.mark.parametrize("model_type", ["dt", "glimmer", "bit"])
def test_groups_cover_each_trainable_parameter_once(model_type: str):
    model = _model(model_type)
    groups = parameter_groups(model, OptimizerConfig())
    assigned = [parameter for group in groups for parameter in group["params"]]

    assert len(assigned) == len({id(parameter) for parameter in assigned})
    assert {id(parameter) for parameter in assigned} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }


def test_bit_policy_is_applied_only_to_bitlinear_master_weights():
    config = OptimizerConfig(
        learning_rate=1e-3,
        weight_decay=0.2,
        bit_learning_rate_multiplier=0.25,
        bit_weight_decay=0.01,
    )
    for model_type in ("dt", "glimmer", "bit"):
        model = _model(model_type)
        groups = {group["group_name"]: group for group in parameter_groups(model, config)}
        if model_type != "bit":
            assert "ternary_master" not in groups
            continue

        expected = {
            id(module.weight) for module in model.modules() if isinstance(module, BitLinear)
        }
        actual = {id(parameter) for parameter in groups["ternary_master"]["params"]}
        assert actual == expected
        assert groups["ternary_master"]["lr"] == pytest.approx(2.5e-4)
        assert groups["ternary_master"]["weight_decay"] == pytest.approx(0.01)


def test_standard_decay_policy_is_explicit_and_optimizer_preserves_names():
    model = _model("dt")
    config = OptimizerConfig(learning_rate=2e-3, weight_decay=0.15)
    optimizer = create_optimizer(model, config)
    groups = {group["group_name"]: group for group in optimizer.param_groups}

    assert set(groups) == {"decay", "no_decay"}
    assert groups["decay"]["lr"] == pytest.approx(2e-3)
    assert groups["decay"]["weight_decay"] == pytest.approx(0.15)
    assert groups["no_decay"]["weight_decay"] == 0.0
    embedding_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    assert embedding_ids <= {id(parameter) for parameter in groups["no_decay"]["params"]}


@pytest.mark.parametrize("model_type", ["dt", "glimmer", "bit"])
def test_optimizer_completes_a_finite_training_update(model_type: str):
    torch.manual_seed(7)
    model = _model(model_type)
    optimizer = create_optimizer(model, OptimizerConfig(learning_rate=1e-3))
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    loss = model(tokens, labels=tokens).loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    optimizer.step()

    assert all(
        torch.isfinite(parameter).all() for parameter in model.parameters()
    )
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )


@pytest.mark.parametrize("field", ["learning_rate", "weight_decay", "epsilon"])
def test_non_finite_optimizer_values_are_rejected(field: str):
    with pytest.raises(ValueError, match="finite"):
        OptimizerConfig(**{field: float("nan")})
