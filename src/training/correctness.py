"""Executable pre-training correctness gates and micro-benchmarks.

These gates answer a narrower question than capability evaluation: can a model
learn, remain causal under adversarial input changes, handle padding, serialize
exactly, resume the same optimization trajectory, and generate a memorized short
sequence? A failure blocks expensive training; a pass is not evidence of quality.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import torch
from torch import Tensor

from src.models import BitConfig, GlimmerConfig, ModelStats, create_model


# Four deterministic, intentionally tiny sequences. Some token contexts conflict,
# so the achievable CE is nonzero; exact prefix continuation is the stronger memory
# check. IDs fit a 32-token synthetic vocabulary and require no tokenizer/data I/O.
TINY_CORPUS = torch.tensor(
    [
        [1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14],
        [1, 4, 8, 12, 16, 20, 24, 28, 3, 7, 11, 15],
        [1, 3, 6, 10, 15, 21, 28, 5, 13, 22, 0, 12],
        [1, 2, 4, 7, 11, 16, 22, 29, 7, 18, 30, 11],
    ],
    dtype=torch.long,
)


@dataclass(frozen=True)
class GateResult:
    model_type: str
    passed: bool
    initial_loss: float
    final_loss: float
    loss_ratio: float
    train_seconds: float
    train_steps_per_second: float
    train_tokens_per_second: float
    causal_max_error: float
    padding_max_error: float
    checkpoint_exact: bool
    deterministic_resume_exact: bool
    generation_exact: bool
    generated_tokens: list[int]
    stats: ModelStats


@dataclass(frozen=True)
class SuiteResult:
    passed: bool
    seed: int
    steps: int
    device: str
    torch_version: str
    python_version: str
    models: tuple[GateResult, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _configs():
    # Same width/depth/head/context/vocab for both models. Intermediate width is
    # fixed rather than parameter-matched because this is a correctness preflight,
    # not the tournament; each model remains small enough for sub-second iteration.
    shared = dict(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )
    return (
        GlimmerConfig(**shared, sliding_window=8),
        BitConfig(**shared),
    )


def _optimizer(model) -> torch.optim.Optimizer:
    # No weight decay removes a confound from a memorization gate. The relatively
    # high LR is intentional for a 120-step preflight and was validated for both.
    return torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)


def _train_step(model, optimizer: torch.optim.Optimizer, batch: Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch, labels=batch).loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("training produced a missing or non-finite loss")
    loss.backward()
    # A shared conservative clip prevents one architecture from passing only by
    # tolerating exploding gradients; it is part of the gate's optimizer contract.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach())


def _causal_error(model, corpus: Tensor) -> float:
    model.eval()
    original = corpus[:1, :8].clone()
    changed = original.clone()
    mutation_index = 5
    changed[0, mutation_index:] = torch.tensor([30, 27, 23], device=corpus.device)
    with torch.inference_mode():
        before = model(original).logits[:, :mutation_index]
        after = model(changed).logits[:, :mutation_index]
    # Earlier logits must be invariant to every modified future token. Exact equality
    # normally holds; a small tolerance admits backend kernel reordering.
    return float((before - after).abs().max())


def _padding_error(model, corpus: Tensor) -> float:
    model.eval()
    unpadded = corpus[:1, :6]
    padded = torch.cat((torch.zeros((1, 2), dtype=torch.long, device=corpus.device), unpadded), dim=1)
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.bool, device=corpus.device)
    with torch.inference_mode():
        reference = model(unpadded).logits
        candidate = model(padded, attention_mask=mask).logits[:, 2:]
    # Position cumsum and key masking should make real-token logits padding invariant.
    return float((reference - candidate).abs().max())


def _nested_equal(left: Any, right: Any) -> bool:
    """Exact recursive comparison for tensor-bearing optimizer state dictionaries."""
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def _checkpoint_exact(model, optimizer: torch.optim.Optimizer, corpus: Tensor, directory: Path) -> bool:
    model.eval()
    expected = model(corpus[:2]).logits.detach().cpu()
    path = directory / f"{model.config.model_type}-roundtrip.pt"
    model.save_checkpoint(path, optimizer=optimizer, step=120, metadata={"gate": "roundtrip"})
    restored, info = type(model).from_checkpoint(path)
    actual = restored(corpus[:2].cpu()).logits.detach().cpu()
    restored_optimizer = _optimizer(restored)
    restored.restore_training_state(path, restored_optimizer, restore_rng=False)
    return (
        info.step == 120
        and info.metadata == {"gate": "roundtrip"}
        and torch.equal(actual, expected)
        and _nested_equal(optimizer.state_dict(), restored_optimizer.state_dict())
    )


def _state_equal(left, right) -> bool:
    left_state, right_state = left.state_dict(), right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[name].cpu(), right_state[name].cpu()) for name in left_state
    )


def _deterministic_resume_exact(config, corpus: Tensor, directory: Path, seed: int) -> bool:
    total_steps, split_step = 12, 6

    def initialize():
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = create_model(config).to(corpus.device)
        return model, _optimizer(model)

    def stochastic_step(model, optimizer):
        # Random batch selection makes RNG restoration observable rather than merely
        # comparing a deterministic zero-dropout optimizer trajectory.
        indices = torch.randint(0, corpus.shape[0], (2,), device=corpus.device)
        _train_step(model, optimizer, corpus[indices])

    uninterrupted, uninterrupted_optimizer = initialize()
    for _ in range(total_steps):
        stochastic_step(uninterrupted, uninterrupted_optimizer)
    uninterrupted_cpu_rng = torch.get_rng_state().clone()
    uninterrupted_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    split, split_optimizer = initialize()
    for _ in range(split_step):
        stochastic_step(split, split_optimizer)
    checkpoint = directory / f"{config.model_type}-resume.pt"
    split.save_checkpoint(checkpoint, split_optimizer, step=split_step, metadata={"gate": "resume"})

    # Fresh construction consumes RNG; restore_training_state must rewind it to the
    # checkpoint value and recover optimizer moments before the remaining steps.
    resumed = create_model(config).to(corpus.device)
    resumed_optimizer = _optimizer(resumed)
    info = resumed.restore_training_state(checkpoint, resumed_optimizer, restore_rng=True)
    for _ in range(split_step, total_steps):
        stochastic_step(resumed, resumed_optimizer)
    rng_equal = torch.equal(uninterrupted_cpu_rng, torch.get_rng_state())
    if uninterrupted_cuda_rng is not None:
        rng_equal = rng_equal and all(
            torch.equal(left, right) for left, right in zip(uninterrupted_cuda_rng, torch.cuda.get_rng_state_all())
        )
    return (
        info.step == split_step
        and _state_equal(uninterrupted, resumed)
        and _nested_equal(uninterrupted_optimizer.state_dict(), resumed_optimizer.state_dict())
        and rng_equal
    )


def _run_model(config, corpus: Tensor, steps: int, seed: int, directory: Path) -> GateResult:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = create_model(config).to(corpus.device)
    optimizer = _optimizer(model)
    model.train()
    started = time.perf_counter()
    initial_loss = _train_step(model, optimizer, corpus)
    final_loss = initial_loss
    for _ in range(1, steps):
        final_loss = _train_step(model, optimizer, corpus)
    train_seconds = time.perf_counter() - started

    prompt = corpus[:1, :4]
    expected = corpus[0].tolist()
    generated = model.generate(prompt, max_new_tokens=len(expected) - prompt.shape[1])[0].tolist()
    causal_error = _causal_error(model, corpus)
    padding_error = _padding_error(model, corpus)
    checkpoint_exact = _checkpoint_exact(model, optimizer, corpus, directory)
    resume_exact = _deterministic_resume_exact(config, corpus, directory, seed + 10_000)
    loss_ratio = final_loss / initial_loss
    generation_exact = generated == expected
    # Thresholds catch broken learning while tolerating normal CPU/GPU numeric drift.
    passed = (
        final_loss < 0.20
        and loss_ratio < 0.10
        and causal_error <= 1e-5
        and padding_error <= 2e-5
        and checkpoint_exact
        and resume_exact
        and generation_exact
    )
    return GateResult(
        model_type=config.model_type,
        passed=passed,
        initial_loss=initial_loss,
        final_loss=final_loss,
        loss_ratio=loss_ratio,
        train_seconds=train_seconds,
        train_steps_per_second=steps / train_seconds,
        train_tokens_per_second=steps * corpus.numel() / train_seconds,
        causal_max_error=causal_error,
        padding_max_error=padding_error,
        checkpoint_exact=checkpoint_exact,
        deterministic_resume_exact=resume_exact,
        generation_exact=generation_exact,
        generated_tokens=generated,
        stats=model.stats(corpus.shape[1]),
    )


def run_correctness_suite(
    steps: int = 120,
    seed: int = 2026,
    device: str | torch.device = "cpu",
) -> SuiteResult:
    """Run every blocking gate for every registered candidate currently in scope."""
    if steps < 2:
        raise ValueError("steps must be at least two")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    # Deterministic kernels are essential because resume equality is a hard gate.
    torch.use_deterministic_algorithms(True)
    corpus = TINY_CORPUS.to(target)
    with tempfile.TemporaryDirectory(prefix="amarken-correctness-") as directory:
        results = tuple(_run_model(config, corpus, steps, seed, Path(directory)) for config in _configs())
    return SuiteResult(
        passed=all(result.passed for result in results),
        seed=seed,
        steps=steps,
        device=str(target),
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        models=results,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_correctness_suite(args.steps, args.seed, args.device)
    text = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(args.output)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
