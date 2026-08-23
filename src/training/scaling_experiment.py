"""Run matched 25M/near-60M DT, Glimmer and Bit experiments across seeds."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time

# CUDA cuBLAS needs a fixed workspace allocation to honor PyTorch's hard
# deterministic-algorithm mode. Set it before importing torch/initializing CUDA;
# :4096:8 is the larger documented deterministic workspace and fits this GPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from src.models import create_config, create_model
from src.tokenization import load_tokenizer, tokenizer_fingerprint
from .data import PackedSequenceDataset
from .bit_optimizer import OptimizerConfig
from .proxy_experiment import _evaluate, _tokenize
from .trainer import Trainer, TrainerConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_model_only(path: Path, trainer: Trainer, seed: int, metadata: dict) -> dict:
    """Retain inference weights without Adam moments that triple disk usage."""
    payload = {
        "format_version": 1,
        "checkpoint_kind": "model_only_scaling",
        "model_type": trainer.model.config.model_type,
        "model_config": asdict(trainer.model.config),
        "model_state": trainer.model.state_dict(),
        "seed": seed,
        "trainer_state": asdict(trainer.state),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "exact_resume": False,
    }


def _rotated_order(seed_index: int) -> tuple[str, ...]:
    names = ("dt", "glimmer", "bit")
    offset = seed_index % len(names)
    return names[offset:] + names[:offset]


def _trajectory_gate(results: list[dict], updates: int) -> tuple[bool, dict[str, int]]:
    """Require matched realized tokens within each seed and exact update counts.

    Packed datasets may contain one deterministic partial final block. Different
    seed shuffles can encounter it a different number of times, so nominal
    batch*context tokens is only an upper bound; architecture fairness requires
    equality among arms sharing a seed, not equality across different shuffles.
    """
    by_seed: dict[str, list[int]] = {}
    for result in results:
        by_seed.setdefault(str(result["seed"]), []).append(result["consumed_tokens"])
    realized = {
        seed: values[0] for seed, values in by_seed.items() if len(set(values)) == 1
    }
    passed = len(realized) == len(by_seed) and all(
        result["optimizer_updates"] == updates for result in results
    )
    return passed, realized


def run(config_path: Path, report_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format_version") != 1:
        raise ValueError("unsupported scaling experiment format")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("configured CUDA device is unavailable")
    torch.use_deterministic_algorithms(True)
    lr_selections = {}
    if config.get("lr_screen_report"):
        lr_report = json.loads(
            Path(config["lr_screen_report"]).read_text(encoding="utf-8")
        )
        if not lr_report.get("passed"):
            raise ValueError("LR-screen report did not pass its token/update gates")
        lr_selections = lr_report["selections"]
    processor = load_tokenizer(config["tokenizer"])
    if processor.vocab_size() != 12_000:
        raise ValueError("scaling tournament requires the fixed 12k tokenizer")
    train_dataset = PackedSequenceDataset(
        _tokenize(Path(config["train_data"]), processor, config["train_token_budget"]),
        config["sequence_length"],
        processor.eos_id(),
        processor.pad_id(),
    )
    validation_dataset = PackedSequenceDataset(
        _tokenize(
            Path(config["validation_data"]),
            processor,
            config["validation_token_budget"],
        ),
        config["sequence_length"],
        processor.eos_id(),
        processor.pad_id(),
    )
    results = []
    for scale, model_configs in config["scales"].items():
        # v2 scales contain named arms; legacy scales map the three model types
        # directly. Normalizing here keeps historical configs executable.
        arms = (
            model_configs.get("variants")
            if "variants" in model_configs
            else {
                name: {"model_type": name, "config": value}
                for name, value in model_configs.items()
            }
        )
        for seed_index, seed in enumerate(config["seeds"]):
            ordered_names = list(arms)
            offset = seed_index % len(ordered_names)
            for variant_name in ordered_names[offset:] + ordered_names[:offset]:
                arm = arms[variant_name]
                model_type = arm["model_type"]
                random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                model_config = create_config(model_type, **arm["config"])
                model = create_model(model_config)
                stats = model.stats(config["sequence_length"])
                run_dir = (
                    Path(config["output_dir"]) / scale / f"seed-{seed}" / variant_name
                )
                learning_rate = arm.get("learning_rate")
                if learning_rate is None:
                    if variant_name in lr_selections:
                        learning_rate = lr_selections[variant_name]["learning_rate"]
                    elif config.get("learning_rate") is not None:
                        # Backward-compatible path for the original shared-LR run.
                        learning_rate = config["learning_rate"]
                    else:
                        raise ValueError(
                            f"no selected learning rate for variant {variant_name!r}"
                        )
                optimizer_config = OptimizerConfig(
                    learning_rate=learning_rate,
                    weight_decay=config["weight_decay"],
                    bit_learning_rate_multiplier=1.0,
                    bit_weight_decay=config["weight_decay"],
                )
                trainer = Trainer(
                    model,
                    train_dataset,
                    TrainerConfig(
                        batch_size=config["batch_size"],
                        gradient_accumulation_steps=config[
                            "gradient_accumulation_steps"
                        ],
                        precision=config["precision"],
                        gradient_checkpointing=config["gradient_checkpointing"],
                        max_grad_norm=config["max_grad_norm"],
                        seed=seed,
                        log_every_steps=1,
                        checkpoint_every_steps=config["optimizer_updates"] + 1,
                        output_dir=run_dir,
                        lr_schedule=config.get("lr_schedule", "constant"),
                        warmup_steps=config.get("warmup_steps", 0),
                        total_steps=config.get(
                            "total_steps", config["optimizer_updates"]
                        ),
                        min_lr_ratio=config.get("min_lr_ratio", 0.1),
                    ),
                    optimizer_config,
                    device,
                )
                initial_validation = _evaluate(
                    model,
                    validation_dataset,
                    device,
                    config["precision"],
                    config["validation_batches"],
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                records = trainer.train(config["optimizer_updates"])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                final_validation = _evaluate(
                    model,
                    validation_dataset,
                    device,
                    config["precision"],
                    config["validation_batches"],
                )
                checkpoint = None
                if seed == config["retain_model_only_seed"]:
                    checkpoint = _save_model_only(
                        run_dir / "final-model-only.pt",
                        trainer,
                        seed,
                        {
                            "scale": scale,
                            "variant": variant_name,
                            "config_sha256": _sha256(config_path),
                            "tokenizer_fingerprint": tokenizer_fingerprint(processor),
                            "tokenizer_kind": processor.kind,
                        },
                    )
                results.append(
                    {
                        "scale": scale,
                        "seed": seed,
                        "variant": variant_name,
                        "model_type": model_type,
                        "learning_rate": learning_rate,
                        "model_config": asdict(model_config),
                        "stats": asdict(stats),
                        "initial_validation_loss": initial_validation,
                        "final_validation_loss": final_validation,
                        "validation_loss_change": final_validation - initial_validation,
                        "first_train_loss": records[0]["loss"],
                        "final_train_loss": records[-1]["loss"],
                        "mean_train_loss": statistics.fmean(
                            record["loss"] for record in records
                        ),
                        "training_seconds": elapsed,
                        "tokens_per_second": trainer.state.consumed_tokens / elapsed,
                        "optimizer_updates": trainer.state.update_step,
                        "consumed_tokens": trainer.state.consumed_tokens,
                        "peak_cuda_allocated_bytes": (
                            torch.cuda.max_memory_allocated()
                            if device.type == "cuda"
                            else None
                        ),
                        "final_gradient_health": {
                            key: value
                            for key, value in records[-1].items()
                            if key.startswith("grad_")
                        },
                        "final_ternary_statistics": {
                            key: value
                            for key, value in records[-1].items()
                            if key.startswith("ternary_")
                        },
                        "checkpoint": checkpoint,
                    }
                )
                print(
                    scale,
                    seed,
                    variant_name,
                    f"val={final_validation:.6f}",
                    f"tok/s={trainer.state.consumed_tokens/elapsed:.1f}",
                    flush=True,
                )
                del trainer, model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    aggregates = {}
    for scale, model_configs in config["scales"].items():
        variant_names = (
            list(model_configs["variants"])
            if "variants" in model_configs
            else list(model_configs)
        )
        for variant_name in variant_names:
            values = [
                result
                for result in results
                if result["scale"] == scale and result["variant"] == variant_name
            ]
            aggregates[f"{scale}:{variant_name}"] = {
                "seeds": [value["seed"] for value in values],
                "final_validation_loss_mean": statistics.fmean(
                    value["final_validation_loss"] for value in values
                ),
                "final_validation_loss_std_population": statistics.pstdev(
                    value["final_validation_loss"] for value in values
                ),
                "validation_improvement_mean": statistics.fmean(
                    -value["validation_loss_change"] for value in values
                ),
                "tokens_per_second_mean": statistics.fmean(
                    value["tokens_per_second"] for value in values
                ),
                "peak_cuda_allocated_bytes_max": max(
                    value["peak_cuda_allocated_bytes"] for value in values
                ),
            }
    expected_tokens = (
        config["optimizer_updates"]
        * config["gradient_accumulation_steps"]
        * config["batch_size"]
        * config["sequence_length"]
    )
    trajectory_passed, realized_tokens = _trajectory_gate(
        results, config["optimizer_updates"]
    )
    report = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "passed": trajectory_passed,
        "interpretation": "three-seed scaling optimization preflight; not a capability ranking",
        "config_sha256": _sha256(config_path),
        "tokenizer_sha256": tokenizer_fingerprint(processor),
        "lr_screen_report_sha256": (
            _sha256(Path(config["lr_screen_report"]))
            if config.get("lr_screen_report")
            else None
        ),
        "train_data_sha256": _sha256(Path(config["train_data"])),
        "validation_data_sha256": _sha256(Path(config["validation_data"])),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "shared": {key: value for key, value in config.items() if key != "scales"},
        "expected_consumed_tokens_per_run": expected_tokens,
        "realized_consumed_tokens_by_seed": realized_tokens,
        "results": results,
        "aggregates": aggregates,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(report_path)
    return report


def audit_existing(report_path: Path) -> dict:
    """Re-evaluate trajectory gates without rerunning completed model training."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    updates = report["shared"]["optimizer_updates"]
    passed, realized = _trajectory_gate(report["results"], updates)
    report["passed"] = passed
    report["realized_consumed_tokens_by_seed"] = realized
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, report_path)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/proxy_scaling.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("experiments/proxy_scaling.json")
    )
    parser.add_argument("--audit-existing", action="store_true")
    args = parser.parse_args()
    if args.audit_existing:
        return 0 if audit_existing(args.report)["passed"] else 1
    return 0 if run(args.config, args.report)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
