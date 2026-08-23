"""Train and evaluate one continuous DT trajectory at cumulative token milestones."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from src.models import create_config, create_model
from src.tokenization import load_tokenizer, tokenizer_fingerprint
from src.training.data import PackedSequenceDataset
from src.training.optimizer import OptimizerConfig
from src.training.proxy_experiment import _evaluate, _tokenize
from src.training.trainer import Trainer, TrainerConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_valid(path: Path, config_sha256: str, tokenizer_sha256: str) -> bool:
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    return (
        metadata.get("learning_curve_config_sha256") == config_sha256
        and metadata.get("tokenizer_fingerprint") == tokenizer_sha256
    )


def run(config_path: Path, report_path: Path, skip_evaluation: bool = False) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format_version") != 1:
        raise ValueError("unsupported learning-curve config")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("configured CUDA device is unavailable")
    torch.use_deterministic_algorithms(True)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    tokenizer = load_tokenizer(config["tokenizer"])
    tokenizer_sha256 = tokenizer_fingerprint(tokenizer)
    config_sha256 = _sha256(config_path)
    if (
        tokenizer.path.suffix != ".json"
        or tokenizer_sha256 != config["tokenizer_fingerprint"]
    ):
        raise ValueError(
            "learning curve is locked to the configured apostrophe-BPE artifact"
        )
    model_config = create_config("dt", **config["model"])
    if model_config.vocab_size != tokenizer.vocab_size():
        raise ValueError("model/tokenizer vocabulary mismatch")
    train = PackedSequenceDataset(
        _tokenize(Path(config["train_data"]), tokenizer, config["train_token_pool"]),
        config["sequence_length"],
        tokenizer.eos_id(),
        tokenizer.pad_id(),
    )
    validation = PackedSequenceDataset(
        _tokenize(
            Path(config["validation_data"]),
            tokenizer,
            config["validation_token_budget"],
        ),
        config["sequence_length"],
        tokenizer.eos_id(),
        tokenizer.pad_id(),
    )
    tokens_per_update = (
        config["batch_size"]
        * config["gradient_accumulation_steps"]
        * config["sequence_length"]
    )
    milestones = config["milestone_tokens"]
    if milestones != sorted(set(milestones)) or any(
        tokens % tokens_per_update for tokens in milestones
    ):
        raise ValueError(
            "milestones must be increasing exact multiples of nominal tokens/update"
        )
    final_updates = milestones[-1] // tokens_per_update
    model = create_model(model_config)
    trainer = Trainer(
        model,
        train,
        TrainerConfig(
            batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            precision=config["precision"],
            gradient_checkpointing=config["gradient_checkpointing"],
            max_grad_norm=config["max_grad_norm"],
            seed=config["seed"],
            log_every_steps=config["log_every_steps"],
            checkpoint_every_steps=final_updates + 1,
            output_dir=Path(config["output_dir"]),
            lr_schedule=config["lr_schedule"],
            warmup_steps=config["warmup_steps"],
            total_steps=final_updates,
            min_lr_ratio=config["min_lr_ratio"],
        ),
        OptimizerConfig(
            learning_rate=config["learning_rate"], weight_decay=config["weight_decay"]
        ),
        device,
    )
    output_dir = Path(config["output_dir"])
    rows = []
    for tokens in milestones:
        target_updates = tokens // tokens_per_update
        checkpoint = output_dir / f"milestone-{tokens}.pt"
        evaluation = output_dir / f"evaluation-v2-{tokens}.json"
        legacy_evaluation = output_dir / f"evaluation-v1-{tokens}.json"
        if _checkpoint_valid(checkpoint, config_sha256, tokenizer_sha256):
            trainer.load_checkpoint(checkpoint)
        elif trainer.state.update_step < target_updates:
            started = time.perf_counter()
            trainer.train(target_updates - trainer.state.update_step)
            training_seconds = time.perf_counter() - started
            trainer.save_checkpoint(
                checkpoint,
                metadata={
                    "variant": "dt-apostrophe-bpe-learning-curve",
                    "milestone_nominal_tokens": tokens,
                    "learning_curve_config_sha256": config_sha256,
                    "tokenizer_fingerprint": tokenizer_sha256,
                    "tokenizer_kind": tokenizer.kind,
                    "training_seconds_segment": training_seconds,
                },
            )
        if trainer.state.update_step != target_updates:
            raise RuntimeError("milestone checkpoint update count mismatch")
        validation_loss = _evaluate(
            trainer.model, validation, device, config["precision"]
        )
        if not skip_evaluation and not evaluation.is_file():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.evaluation.meaningful_scale",
                    "--checkpoint",
                    str(checkpoint),
                    "--tokenizer",
                    config["tokenizer"],
                    "--benchmark",
                    config["benchmark"],
                    "--report",
                    str(evaluation),
                    "--device",
                    config["evaluation_device"],
                ],
                check=True,
            )
        if not skip_evaluation and not legacy_evaluation.is_file():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.evaluation.capability",
                    "--worker",
                    "--checkpoint",
                    str(checkpoint),
                    "--tokenizer",
                    config["tokenizer"],
                    "--benchmark",
                    config["legacy_benchmark"],
                    "--validation",
                    config["validation_data"],
                    "--worker-output",
                    str(legacy_evaluation),
                    "--device",
                    config["evaluation_device"],
                ],
                check=True,
            )
        rows.append(
            {
                "milestone_nominal_tokens": tokens,
                "optimizer_updates": target_updates,
                "realized_consumed_tokens": trainer.state.consumed_tokens,
                "packed_data_epochs_completed": trainer.state.epoch,
                "nominal_corpus_passes": tokens / config["train_token_pool"],
                "validation_loss": validation_loss,
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": _sha256(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                },
                "evaluation_v2": str(evaluation) if evaluation.is_file() else None,
                "evaluation_v1": (
                    str(legacy_evaluation) if legacy_evaluation.is_file() else None
                ),
            }
        )
    report = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "passed": len(rows) == len(milestones),
        "interpretation": "one continuous control-model learning curve; milestones are not independent restarts",
        "config_sha256": config_sha256,
        "tokenizer_fingerprint": tokenizer_sha256,
        "train_data_sha256": _sha256(Path(config["train_data"])),
        "validation_data_sha256": _sha256(Path(config["validation_data"])),
        "benchmark_sha256": _sha256(Path(config["benchmark"])),
        "legacy_benchmark_sha256": _sha256(Path(config["legacy_benchmark"])),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": config["device"],
        },
        "model_config": asdict(model_config),
        "tokens_per_update_nominal": tokens_per_update,
        "train_token_pool": config["train_token_pool"],
        "milestones": rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(report_path)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/learning_curve_dt_apostrophe_100m.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/learning_curve_dt_apostrophe_100m.json"),
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    report = run(args.config, args.report, args.skip_evaluation)
    print(
        json.dumps(
            {"passed": report["passed"], "milestones": report["milestones"]}, indent=2
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
