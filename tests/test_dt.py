import torch

from src.models.dt import DTCausalLM, DTConfig


def tiny_config(**overrides):
    values = dict(
        vocab_size=97, hidden_size=32, intermediate_size=72,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=64,
    )
    values.update(overrides)
    return DTConfig(**values)


def test_dt_forward_backward_tying_and_padding_safety():
    model = DTCausalLM(tiny_config())
    tokens = torch.randint(0, model.config.vocab_size, (2, 7))
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1]])
    labels = tokens.masked_fill(~mask.bool(), -100)
    output = model(tokens, attention_mask=mask, labels=labels)
    assert output.logits.shape == (2, 7, model.config.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert torch.isfinite(output.logits).all()
    assert model.lm_head.weight is model.token_embedding.weight
    output.loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_dt_default_model_stays_below_project_limit():
    assert DTCausalLM().parameter_count() <= 60_000_000
