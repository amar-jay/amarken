from src.training.scaling_experiment import _trajectory_gate


def test_trajectory_gate_matches_realized_tokens_within_seed_not_across_seeds():
    rows = [
        {"seed": 1, "consumed_tokens": 100, "optimizer_updates": 5},
        {"seed": 1, "consumed_tokens": 100, "optimizer_updates": 5},
        {"seed": 2, "consumed_tokens": 96, "optimizer_updates": 5},
        {"seed": 2, "consumed_tokens": 96, "optimizer_updates": 5},
    ]
    passed, realized = _trajectory_gate(rows, updates=5)
    assert passed and realized == {"1": 100, "2": 96}
    rows[-1]["consumed_tokens"] = 95
    assert not _trajectory_gate(rows, updates=5)[0]
