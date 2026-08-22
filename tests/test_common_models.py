from pathlib import Path

import torch

from src.models import (
    AmarkenCausalLM,
    BitCausalLM,
    BitConfig,
    CausalLMOutput,
    GlimmerCausalLM,
    GlimmerConfig,
    DTCausalLM,
    DTConfig,
    create_config,
    create_model,
    load_config,
    save_config,
)


def configs():
    return (
        DTConfig(
            vocab_size=67,
            hidden_size=32,
            intermediate_size=72,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
        ),
        GlimmerConfig(
            vocab_size=67,
            hidden_size=32,
            intermediate_size=72,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            sliding_window=4,
        ),
        BitConfig(
            vocab_size=67,
            hidden_size=32,
            intermediate_size=72,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
        ),
    )


def test_factory_and_forward_contract_are_identical():
    for config in configs():
        model = create_model(config)
        assert isinstance(model, AmarkenCausalLM)
        tokens = torch.randint(0, config.vocab_size, (2, 7))
        output = model(tokens, labels=tokens)
        assert isinstance(output, CausalLMOutput)
        assert output.logits.shape == (2, 7, config.vocab_size)
        assert output.loss is not None and torch.isfinite(output.loss)


def test_registry_config_json_round_trip(tmp_path: Path):
    for config in configs():
        path = tmp_path / f"{config.model_type}.json"
        save_config(config, path)
        restored = load_config(path)
        assert restored == config
        assert create_config(config.model_type, **config.to_dict()) == config


def test_shared_stats_schema_is_architecture_aware():
    dt, glimmer, bit = (create_model(config) for config in configs())
    for model in (dt, glimmer, bit):
        stats = model.stats(sequence_length=16)
        assert stats.total_parameters == model.parameter_count()
        assert stats.active_parameters == stats.total_parameters
        assert stats.forward_flops > 0 and stats.flops_per_token == stats.forward_flops / 16
        assert stats.artifact_bytes > 0 and stats.kv_cache_bytes > 0
    assert dt.stats(16).ternary_parameters == glimmer.stats(16).ternary_parameters == 0
    assert bit.stats(16).ternary_parameters > 0
    assert bit.stats(16).artifact_bytes < bit.stats(16).training_parameter_bytes


def test_generation_contract_and_mode_restoration():
    for config in configs():
        model = create_model(config)
        model.train()
        prompt = torch.randint(0, config.vocab_size, (2, 5))
        generated = model.generate(prompt, max_new_tokens=3)
        assert generated.shape == (2, 8)
        assert torch.equal(generated[:, :5], prompt)
        assert model.training


def test_checkpoint_model_optimizer_metadata_and_rng_round_trip(tmp_path: Path):
    for config in configs():
        torch.manual_seed(123)
        model = create_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tokens = torch.randint(0, config.vocab_size, (2, 6))
        loss = model(tokens, labels=tokens).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        expected = model(tokens).logits
        path = tmp_path / f"{config.model_type}.pt"
        model.save_checkpoint(path, optimizer=optimizer, step=17, metadata={"seed": 123})

        model_class = {"dt": DTCausalLM, "glimmer": GlimmerCausalLM, "bit": BitCausalLM}[config.model_type]
        restored, info = model_class.from_checkpoint(path)
        assert info.step == 17 and info.metadata == {"seed": 123}
        assert torch.equal(restored(tokens).logits, expected)

        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        resume_info = restored.restore_training_state(path, restored_optimizer, restore_rng=True)
        assert resume_info == info
        assert restored_optimizer.state_dict()["state"]
