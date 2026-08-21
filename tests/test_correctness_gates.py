from src.training.correctness import run_correctness_suite


def test_full_correctness_suite_passes_both_models():
    result = run_correctness_suite(steps=120, seed=2026, device="cpu")
    assert result.passed
    assert {model.model_type for model in result.models} == {"glimmer", "bit"}
    for model in result.models:
        assert model.passed
        assert model.final_loss < 0.20
        assert model.loss_ratio < 0.10
        assert model.causal_max_error <= 1e-5
        assert model.padding_max_error <= 2e-5
        assert model.checkpoint_exact
        assert model.deterministic_resume_exact
        assert model.generation_exact
