import torch

from src.models.glimmer import GlimmerCausalLM, GlimmerConfig


def tiny_config(**overrides):
    values = dict(
        vocab_size=101,
        hidden_size=32,
        intermediate_size=80,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=4,
    )
    values.update(overrides)
    return GlimmerConfig(**values)


def test_layer_pattern_is_anchored_on_final_global_layer():
    assert tiny_config().layer_types == (
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    )


def test_forward_loss_backward_and_tied_weights():
    torch.manual_seed(0)
    model = GlimmerCausalLM(tiny_config())
    tokens = torch.randint(0, model.config.vocab_size, (2, 9))
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (2, 9, model.config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert model.lm_head.weight is model.token_embedding.weight
    output.loss.backward()
    assert model.layers[0].attention.query_scale.grad is not None


def test_padding_mask_does_not_produce_nan():
    model = GlimmerCausalLM(tiny_config())
    tokens = torch.randint(0, model.config.vocab_size, (2, 7))
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]])
    assert torch.isfinite(model(tokens, attention_mask=mask).logits).all()


def test_left_and_complete_padding_have_finite_outputs_and_gradients():
    model = GlimmerCausalLM(tiny_config())
    tokens = torch.randint(0, model.config.vocab_size, (2, 7))
    # These produce fully masked rows under column-only padding: left padding in
    # example zero and the stronger entirely padded case in example one.
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]])
    labels = tokens.masked_fill(~mask.bool(), -100)
    output = model(tokens, attention_mask=mask, labels=labels)
    assert torch.isfinite(output.logits).all()
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_masks_restore_padded_query_diagonals():
    model = GlimmerCausalLM(tiny_config())
    padding = torch.tensor([[0, 0, 1, 1]])
    masks = model._attention_masks(padding, batch=1, length=4, device=padding.device)
    for mask in masks.values():
        assert mask.shape == (1, 1, 4, 4)
        assert mask[0, 0, 0, 0] == 0 and mask[0, 0, 1, 1] == 0
        assert not torch.isneginf(mask).all(dim=-1).any()


def test_layers_reuse_only_two_mask_objects_per_forward():
    model = GlimmerCausalLM(tiny_config())
    tokens = torch.randint(0, model.config.vocab_size, (1, 7))
    seen_mask_ids = []
    hooks = [
        layer.attention.register_forward_pre_hook(lambda _module, args: seen_mask_ids.append(id(args[2])))
        for layer in model.layers
    ]
    model(tokens)
    for hook in hooks:
        hook.remove()
    assert len(seen_mask_ids) == model.config.num_hidden_layers
    assert len(set(seen_mask_ids)) == 2


def test_default_config_is_created_per_model_instance():
    assert GlimmerCausalLM().config is not GlimmerCausalLM().config


def test_architecture_ablation_switches_remove_only_requested_mechanisms():
    model = GlimmerCausalLM(tiny_config(use_attention_gate=False, use_qk_norm=False, use_nope_global=False))
    attention = model.layers[0].attention
    assert attention.gate_proj is None
    assert attention.qk_norm is None and attention.query_scale is None
    tokens = torch.randint(0, model.config.vocab_size, (1, 7))
    assert torch.isfinite(model(tokens).logits).all()


def test_default_model_stays_below_project_limit():
    model = GlimmerCausalLM()
    assert model.parameter_count() <= 60_000_000
