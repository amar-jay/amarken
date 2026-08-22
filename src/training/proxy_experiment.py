"""Run a matched DT/Glimmer/Bit smoke-scale experiment on proxy-v1."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time

import sentencepiece as spm
import torch

from src.models import create_config, create_model
from .data import PackedSequenceDataset, TokenizedExample
from .optimizer import OptimizerConfig
from .trainer import Trainer, TrainerConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokenize(path: Path, processor: spm.SentencePieceProcessor, token_budget: int) -> list[TokenizedExample]:
    """Read stable JSONL order until exactly `token_budget` source tokens exist."""
    examples, consumed = [], 0
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            text = json.loads(line)["text"]
            ids = processor.encode(text, out_type=int)
            remaining = token_budget - consumed
            if not ids or remaining <= 0:
                break
            ids = ids[:remaining]
            # This is source-text pretraining rather than chat SFT, so every real
            # source token is eligible. The shared packer still applies boundary,
            # EOS and padding masks; no teacher/distillation targets are present.
            examples.append(TokenizedExample(tuple(ids), tuple(True for _ in ids)))
            consumed += len(ids)
            if consumed == token_budget:
                break
    if consumed != token_budget:
        raise ValueError(f"{path} supplied {consumed}, expected {token_budget} tokenizer tokens")
    return examples


@torch.no_grad()
def _evaluate(model, dataset: PackedSequenceDataset, device: torch.device, precision: str, batches: int = 32) -> float:
    model.eval()
    weighted_loss = targets = 0
    autocast = nullcontext if precision == "fp32" else lambda: torch.autocast(
        device_type=device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16
    )
    for start in range(0, min(len(dataset), batches)):
        batch = dataset.batch([start], device)
        count = int((batch["labels"][:, 1:] != -100).sum())
        if not count:
            continue
        with autocast():
            loss = model(**batch).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("evaluation produced non-finite loss")
        weighted_loss += float(loss) * count
        targets += count
    model.train()
    return weighted_loss / targets


def run(config_path: Path, report_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format_version") != 1:
        raise ValueError("unsupported proxy experiment format")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("configured CUDA device is unavailable")
    # Hard determinism makes seed comparisons and interrupted resumes auditable.
    # PyTorch selects the deterministic SDPA backend where the active device needs it.
    torch.use_deterministic_algorithms(True)
    tokenizer_path = Path(config["tokenizer"])
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if processor.vocab_size() != 12_000:
        raise ValueError("proxy tournament requires the selected matched 12k tokenizer")
    train_examples = _tokenize(Path(config["train_data"]), processor, config["train_token_budget"])
    validation_examples = _tokenize(Path(config["validation_data"]), processor, config["validation_token_budget"])
    train_dataset = PackedSequenceDataset(train_examples, config["sequence_length"], processor.eos_id(), processor.pad_id())
    validation_dataset = PackedSequenceDataset(validation_examples, config["sequence_length"], processor.eos_id(), processor.pad_id())
    optimizer_config = OptimizerConfig(
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
        # Equal decay/LR makes Bit's model-specific group structurally distinct but
        # numerically matched to full-precision projection groups in this control.
        bit_learning_rate_multiplier=1.0, bit_weight_decay=config["weight_decay"],
    )
    output_root = Path(config["output_dir"])
    results = []
    for model_type in ("dt", "glimmer", "bit"):
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
        model_config = create_config(model_type, **config["models"][model_type])
        model = create_model(model_config)
        stats = model.stats(config["sequence_length"])
        trainer_config = TrainerConfig(
            batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            precision=config["precision"], gradient_checkpointing=config["gradient_checkpointing"],
            max_grad_norm=config["max_grad_norm"], seed=config["seed"], log_every_steps=1,
            checkpoint_every_steps=config["optimizer_updates"] + 1,
            output_dir=output_root / model_type,
            # Optional keys preserve old smoke configs while letting serious runs
            # declare the complete trajectory in their provenance JSON.
            lr_schedule=config.get("lr_schedule", "constant"),
            warmup_steps=config.get("warmup_steps", 0),
            total_steps=config.get("total_steps", config["optimizer_updates"]),
            min_lr_ratio=config.get("min_lr_ratio", 0.1),
        )
        trainer = Trainer(model, train_dataset, trainer_config, optimizer_config, device)
        initial_validation_loss = _evaluate(model, validation_dataset, device, config["precision"])
        started = time.perf_counter()
        records = trainer.train(config["optimizer_updates"])
        training_seconds = time.perf_counter() - started
        final_validation_loss = _evaluate(model, validation_dataset, device, config["precision"])
        final_checkpoint = output_root / model_type / "final.pt"
        trainer.save_checkpoint(final_checkpoint)
        results.append({
            "model_type": model_type, "model_config": asdict(model_config), "stats": asdict(stats),
            "initial_validation_loss": initial_validation_loss,
            "final_validation_loss": final_validation_loss,
            "validation_loss_change": final_validation_loss - initial_validation_loss,
            "first_train_loss": records[0]["loss"], "final_train_loss": records[-1]["loss"],
            "mean_train_loss": statistics.fmean(record["loss"] for record in records),
            "training_seconds": training_seconds,
            "tokens_per_second": trainer.state.consumed_tokens / training_seconds,
            "optimizer_updates": trainer.state.update_step,
            "consumed_tokens": trainer.state.consumed_tokens,
            "final_gradient_health": {key: value for key, value in records[-1].items() if key.startswith("grad_")},
            "final_ternary_statistics": {key: value for key, value in records[-1].items() if key.startswith("ternary_")},
            "checkpoint": {"path": str(final_checkpoint), "sha256": _sha256(final_checkpoint), "bytes": final_checkpoint.stat().st_size},
        })
    # Equality gates turn accidental per-arm changes into a failed experiment,
    # rather than a misleading comparison table.
    matched = len({result["consumed_tokens"] for result in results}) == 1 and len({result["optimizer_updates"] for result in results}) == 1
    report = {
        "format_version": 1, "experiment_id": config["experiment_id"], "passed": matched,
        "interpretation": "smoke-scale optimization comparison; not a capability ranking",
        "config_sha256": _sha256(config_path), "tokenizer_sha256": _sha256(tokenizer_path),
        "train_data_sha256": _sha256(Path(config["train_data"])),
        "validation_data_sha256": _sha256(Path(config["validation_data"])),
        "torch_version": torch.__version__, "sentencepiece_version": spm.__version__,
        "python_version": platform.python_version(), "shared": {key: value for key, value in config.items() if key not in {"models"}},
        "train_blocks": len(train_dataset), "validation_blocks": len(validation_dataset),
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/proxy_10m.json"))
    parser.add_argument("--report", type=Path, default=Path("experiments/proxy_10m.json"))
    args = parser.parse_args()
    report = run(args.config, args.report)
    for result in report["results"]:
        print(result["model_type"], result["stats"]["total_parameters"], result["final_train_loss"], result["final_validation_loss"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
