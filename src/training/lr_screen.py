"""Deterministically screen architecture-specific learning rates before long runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

# Deterministic CUDA matrix multiplication requires this to exist before torch
# initializes CUDA; the larger workspace is negligible beside model activations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from src.models import create_config, create_model
from src.tokenization import load_tokenizer
from .data import PackedSequenceDataset
from .optimizer import OptimizerConfig
from .proxy_experiment import _evaluate, _tokenize
from .trainer import Trainer, TrainerConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(config_path: Path, report_path: Path) -> dict:
    """Train every variant/LR pair with identical tokens and select by val loss."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format_version") != 1:
        raise ValueError("unsupported LR-screen config format")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("configured CUDA device is unavailable")
    torch.use_deterministic_algorithms(True)
    processor = load_tokenizer(config["tokenizer"])
    train = PackedSequenceDataset(
        _tokenize(Path(config["train_data"]), processor, config["train_token_budget"]),
        config["sequence_length"], processor.eos_id(), processor.pad_id(),
    )
    validation = PackedSequenceDataset(
        _tokenize(Path(config["validation_data"]), processor, config["validation_token_budget"]),
        config["sequence_length"], processor.eos_id(), processor.pad_id(),
    )
    results = []
    # Variant then LR ordering makes interruptions auditable and comparisons stable.
    for variant_name, arm in config["variants"].items():
        for learning_rate in config["learning_rates"]:
            seed = config["seed"]
            random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model_config = create_config(arm["model_type"], **arm["config"])
            model = create_model(model_config)
            run_name = f"{variant_name}-lr-{learning_rate:.3g}"
            trainer = Trainer(
                model, train,
                TrainerConfig(
                    batch_size=config["batch_size"],
                    gradient_accumulation_steps=config["gradient_accumulation_steps"],
                    precision=config["precision"],
                    gradient_checkpointing=config["gradient_checkpointing"],
                    max_grad_norm=config["max_grad_norm"], seed=seed,
                    log_every_steps=config["optimizer_updates"],
                    checkpoint_every_steps=config["optimizer_updates"] + 1,
                    output_dir=Path(config["output_dir"]) / run_name,
                    lr_schedule="cosine", warmup_steps=config["warmup_steps"],
                    total_steps=config["optimizer_updates"], min_lr_ratio=config["min_lr_ratio"],
                ),
                OptimizerConfig(
                    learning_rate=learning_rate, weight_decay=config["weight_decay"],
                    bit_learning_rate_multiplier=1.0, bit_weight_decay=config["weight_decay"],
                ), device,
            )
            initial = _evaluate(model, validation, device, config["precision"], config["validation_batches"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            records = trainer.train(config["optimizer_updates"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            final = _evaluate(model, validation, device, config["precision"], config["validation_batches"])
            results.append({
                "variant": variant_name, "model_type": arm["model_type"],
                "model_config": asdict(model_config), "learning_rate": learning_rate,
                "initial_validation_loss": initial, "final_validation_loss": final,
                "validation_improvement": initial - final,
                "final_train_loss": records[-1]["loss"],
                "consumed_tokens": trainer.state.consumed_tokens,
                "optimizer_updates": trainer.state.update_step,
                "training_seconds": elapsed,
                "tokens_per_second": trainer.state.consumed_tokens / elapsed,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                "final_gradient_health": {k: v for k, v in records[-1].items() if k.startswith("grad_")},
                "final_ternary_statistics": {k: v for k, v in records[-1].items() if k.startswith("ternary_")},
            })
            del trainer, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    selections = {}
    for variant_name in config["variants"]:
        candidates = [row for row in results if row["variant"] == variant_name]
        winner = min(candidates, key=lambda row: (row["final_validation_loss"], row["learning_rate"]))
        selections[variant_name] = {
            "learning_rate": winner["learning_rate"],
            "final_validation_loss": winner["final_validation_loss"],
            "median_candidate_loss": statistics.median(row["final_validation_loss"] for row in candidates),
        }
    expected = config["optimizer_updates"] * config["gradient_accumulation_steps"] * config["batch_size"] * config["sequence_length"]
    report = {
        "format_version": 1, "experiment_id": config["experiment_id"],
        "passed": all(row["consumed_tokens"] == expected for row in results),
        "interpretation": "short LR selection only; not an architecture ranking",
        "config_sha256": _sha256(config_path), "expected_tokens_per_candidate": expected,
        "selections": selections, "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lr_screen_10m.json"))
    parser.add_argument("--report", type=Path, default=Path("experiments/lr_screen_10m.json"))
    args = parser.parse_args()
    report = run(args.config, args.report)
    for variant, selection in report["selections"].items():
        print(variant, selection["learning_rate"], selection["final_validation_loss"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
