import math

import torch

from src.models.bit import BitCausalLM, BitConfig, BitLinear, pack_ternary


def tiny_config(**overrides):
    values = dict(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=72,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    values.update(overrides)
    return BitConfig(**values)


def test_bitlinear_forward_is_ternary_and_ste_trains_master():
    layer = BitLinear(5, 3, eps=1e-5)
    torch.nn.init.normal_(layer.weight)
    trits, scale = layer.quantized()
    assert set(trits.unique().tolist()) <= {-1, 0, 1}
    x = torch.randn(2, 5, requires_grad=True)
    expected = torch.nn.functional.linear(x, trits.float() * scale)
    actual = layer(x)
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.sum().backward()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()


def test_pack_ternary_uses_four_trits_per_byte():
    trits = torch.tensor([-1, 0, 1, -1, 1])
    packed, padding = pack_ternary(trits)
    assert packed.dtype == torch.uint8
    assert packed.numel() == math.ceil(trits.numel() / 4)
    assert padding == 3


def test_output_channel_scaling_has_one_scale_per_neuron():
    layer = BitLinear(5, 3, eps=1e-5, scale_granularity="output_channel")
    torch.nn.init.normal_(layer.weight)
    trits, scale = layer.quantized()
    assert scale.shape == (3, 1)
    inputs = torch.randn(2, 5)
    assert torch.allclose(
        layer(inputs),
        torch.nn.functional.linear(inputs, trits.float() * scale),
        atol=1e-6,
    )


def test_forward_loss_backward_and_tied_embeddings():
    model = BitCausalLM(tiny_config())
    tokens = torch.randint(0, model.config.vocab_size, (2, 9))
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (2, 9, model.config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert model.lm_head.weight is model.token_embedding.weight
    output.loss.backward()
    assert model.layers[0].attention.q_proj.weight.grad is not None


def test_left_and_complete_padding_remain_finite():
    model = BitCausalLM(tiny_config())
    tokens = torch.randint(0, model.config.vocab_size, (2, 7))
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]])
    labels = tokens.masked_fill(~mask.bool(), -100)
    output = model(tokens, attention_mask=mask, labels=labels)
    assert torch.isfinite(output.logits).all() and torch.isfinite(output.loss)
    output.loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_artifact_accounting_and_export_cover_every_bitlinear():
    model = BitCausalLM(tiny_config())
    report = model.artifact_report()
    modules = sum(isinstance(module, BitLinear) for module in model.modules())
    assert report.total_parameters == model.parameter_count()
    assert report.ternary_parameters > report.floating_parameters
    assert (
        report.theoretical_bytes
        <= report.packed_2bit_bytes
        < report.training_master_bytes_fp32
    )
    assert len(model.export_ternary()) == modules


def test_default_model_stays_below_project_limit():
    model = BitCausalLM()
    assert model.parameter_count() <= 60_000_000
