from pathlib import Path

import torch

from src.models import BitConfig, GlimmerConfig, create_model
from src.training import PackedSequenceDataset, TokenizedExample, Trainer, TrainerConfig
from src.training.optimizer import OptimizerConfig, parameter_groups


def _dataset() -> PackedSequenceDataset:
    examples = [
        TokenizedExample((4, 5, 6, 7), (False, False, True, True)),
        TokenizedExample((8, 9, 10), (False, True, True)),
        TokenizedExample((11, 12, 13, 14), (True, True, True, True)),
        TokenizedExample((15, 16, 17), (False, True, True)),
    ]
    return PackedSequenceDataset(examples, sequence_length=8, eos_token_id=2, pad_token_id=3)


def _glimmer_config() -> GlimmerConfig:
    return GlimmerConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=16, sliding_window=4,
    )


def _bit_config() -> BitConfig:
    return BitConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=16,
    )


def _nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def test_packing_masks_nonassistant_padding_eos_and_record_boundaries():
    dataset = _dataset()
    assert all(len(block.input_ids) == 8 for block in dataset)
    assert all(block.labels[0] == -100 for block in dataset)
    assert all(segment == -1 for block in dataset for segment, valid in zip(block.segment_ids, block.attention_mask) if not valid)
    assert all(label == -100 for block in dataset for label, valid in zip(block.labels, block.attention_mask) if not valid)
    flattened_labels = [label for block in dataset for label in block.labels]
    assert 2 not in flattened_labels  # inserted EOS is context, never a target
    # Token 11 begins the third record with assistant_mask=True but must not be
    # learned from the preceding packed record's context.
    assert all(not (token == 11 and label == 11) for block in dataset for token, label in zip(block.input_ids, block.labels))


def test_warmup_cosine_schedule_is_logged_at_optimizer_boundaries(tmp_path: Path):
    config = TrainerConfig(
        batch_size=1, gradient_accumulation_steps=2, precision="fp32", gradient_checkpointing=False,
        log_every_steps=1, checkpoint_every_steps=99, output_dir=tmp_path,
        warmup_steps=2, total_steps=4, lr_schedule="cosine", min_lr_ratio=0.1,
    )
    trainer = Trainer(create_model(_glimmer_config()), _dataset(), config)
    records = trainer.train(4)
    rates = [record["learning_rates"]["decay"] for record in records]
    assert torch.allclose(torch.tensor(rates), torch.tensor([1.5e-4, 3e-4, 1.65e-4, 3e-5]))


def test_optimizer_groups_are_model_specific_and_complete():
    ordinary = create_model(_glimmer_config())
    bit = create_model(_bit_config())
    ordinary_groups = parameter_groups(ordinary, OptimizerConfig())
    bit_groups = parameter_groups(bit, OptimizerConfig())
    assert {group["group_name"] for group in ordinary_groups} == {"decay", "no_decay"}
    # Bit's only FP matrix is the tied embedding/head and is intentionally exempt
    # from decay, so its active groups are norms/embedding plus ternary masters.
    assert {group["group_name"] for group in bit_groups} == {"no_decay", "ternary_master"}
    for model, groups in ((ordinary, ordinary_groups), (bit, bit_groups)):
        assert sum(parameter.numel() for group in groups for parameter in group["params"]) == model.parameter_count(True)


def test_exact_resume_with_accumulation_and_checkpointing(tmp_path: Path):
    config = TrainerConfig(
        batch_size=1, gradient_accumulation_steps=2, precision="fp32",
        gradient_checkpointing=True, log_every_steps=1, checkpoint_every_steps=99,
        output_dir=tmp_path / "continuous", seed=77,
        lr_schedule="cosine", warmup_steps=1, total_steps=4, min_lr_ratio=0.1,
    )
    torch.manual_seed(123)
    continuous = Trainer(create_model(_glimmer_config()), _dataset(), config)
    continuous.train(4)

    torch.manual_seed(123)
    split_config = TrainerConfig(**{**config.__dict__, "output_dir": tmp_path / "split"})
    split = Trainer(create_model(_glimmer_config()), _dataset(), split_config)
    split.train(2)
    checkpoint = tmp_path / "resume.pt"
    split.save_checkpoint(checkpoint)

    torch.manual_seed(999)  # Loading must replace model and RNG initialization.
    resumed = Trainer(create_model(_glimmer_config()), _dataset(), split_config)
    resumed.load_checkpoint(checkpoint)
    resumed.train(2)
    assert resumed.state == continuous.state
    assert _nested_equal(resumed.model.state_dict(), continuous.model.state_dict())
    assert _nested_equal(resumed.optimizer.state_dict(), continuous.optimizer.state_dict())

    incompatible_config = TrainerConfig(**{**split_config.__dict__, "seed": 78})
    incompatible = Trainer(create_model(_glimmer_config()), _dataset(), incompatible_config)
    try:
        incompatible.load_checkpoint(checkpoint)
    except ValueError as error:
        assert "trajectory configuration" in str(error)
    else:
        raise AssertionError("resume accepted a changed shuffle seed")


def test_bit_logs_ternary_and_gradient_health(tmp_path: Path):
    config = TrainerConfig(
        batch_size=1, precision="fp32", gradient_checkpointing=False,
        log_every_steps=1, checkpoint_every_steps=99, output_dir=tmp_path,
    )
    trainer = Trainer(create_model(_bit_config()), _dataset(), config)
    records = trainer.train(1)
    assert len(records) == 1
    record = records[0]
    assert 0 <= record["ternary_zero_fraction"] <= 1
    assert record["ternary_scale_min"] > 0
    assert record["grad_finite_fraction"] == 1
    assert record["ternary_grad_finite_fraction"] == 1
    assert (tmp_path / "metrics.jsonl").is_file()
