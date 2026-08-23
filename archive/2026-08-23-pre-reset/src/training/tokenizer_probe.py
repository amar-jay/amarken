"""Matched downstream DT probe for qualified tokenizer candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch

from src.tokenization import (
    load_tokenizer,
    tokenizer_artifact_bytes,
    tokenizer_fingerprint,
)
from src.training.proxy_experiment import run as run_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_exposure(path: Path, tokenizer, token_budget: int) -> dict:
    """Measure source coverage needed to supply an exact tokenizer-token budget."""
    tokens = utf8_bytes = characters = documents = 0
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            text = json.loads(line)["text"]
            ids = tokenizer.encode(text)
            remaining = token_budget - tokens
            if remaining <= 0:
                break
            documents += 1
            if len(ids) <= remaining:
                tokens += len(ids)
                utf8_bytes += len(text.encode("utf-8"))
                characters += len(text)
            else:
                # The trainer consumes this fraction of the final document's
                # tokenization. Source coverage is reported as an explicit
                # proportional estimate because byte offsets are tokenizer-specific.
                fraction = remaining / len(ids)
                tokens += remaining
                utf8_bytes += round(len(text.encode("utf-8")) * fraction)
                characters += round(len(text) * fraction)
                break
    if tokens != token_budget:
        raise ValueError(
            f"{path} supplied {tokens}, expected {token_budget} tokenizer tokens"
        )
    return {
        "tokenizer_tokens": tokens,
        "estimated_utf8_bytes": utf8_bytes,
        "estimated_characters": characters,
        "documents_touched": documents,
        "estimation": "complete documents plus proportional final-document coverage",
    }


def run(config_path: Path, report_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    tokenizer_metadata = {}
    for candidate in config["candidates"]:
        tokenizer = load_tokenizer(candidate["path"])
        if tokenizer.vocab_size() != config["model"]["vocab_size"]:
            raise ValueError(f"{candidate['name']} vocabulary does not match the model")
        fingerprint = tokenizer_fingerprint(tokenizer)
        tokenizer_metadata[candidate["name"]] = {
            "path": candidate["path"],
            "kind": tokenizer.kind,
            "vocab_size": tokenizer.vocab_size(),
            "artifact_bytes": tokenizer_artifact_bytes(tokenizer),
            "fingerprint": fingerprint,
            "train_source_exposure": _source_exposure(
                Path(config["train_data"]), tokenizer, config["train_token_budget"]
            ),
            "validation_source_exposure": _source_exposure(
                Path(config["validation_data"]),
                tokenizer,
                config["validation_token_budget"],
            ),
        }
        for seed in config["seeds"]:
            for learning_rate in config["learning_rates"]:
                arm_id = f"{candidate['name']}-seed-{seed}-lr-{learning_rate:g}"
                arm_dir = root / arm_id
                arm_config = {
                    "format_version": 1,
                    "experiment_id": f"{config['experiment_id']}:{arm_id}",
                    "seed": seed,
                    "device": config["device"],
                    "precision": config["precision"],
                    "sequence_length": config["sequence_length"],
                    "train_token_budget": config["train_token_budget"],
                    "validation_token_budget": config["validation_token_budget"],
                    "batch_size": config["batch_size"],
                    "gradient_accumulation_steps": config[
                        "gradient_accumulation_steps"
                    ],
                    "optimizer_updates": config["optimizer_updates"],
                    "learning_rate": learning_rate,
                    "weight_decay": config["weight_decay"],
                    "gradient_checkpointing": config["gradient_checkpointing"],
                    "max_grad_norm": config["max_grad_norm"],
                    "tokenizer": candidate["path"],
                    "train_data": config["train_data"],
                    "validation_data": config["validation_data"],
                    "output_dir": str(arm_dir / "checkpoint"),
                    "models": {"dt": config["model"]},
                }
                arm_config_path = arm_dir / "config.json"
                arm_report_path = arm_dir / "report.json"
                arm_dir.mkdir(parents=True, exist_ok=True)
                arm_config_path.write_text(
                    json.dumps(arm_config, indent=2, sort_keys=True) + "\n"
                )
                if arm_report_path.is_file():
                    cached = json.loads(arm_report_path.read_text(encoding="utf-8"))
                    arm_report = (
                        cached
                        if cached.get("config_sha256") == _sha256(arm_config_path)
                        else run_model(arm_config_path, arm_report_path)
                    )
                else:
                    arm_report = run_model(arm_config_path, arm_report_path)
                result = arm_report["results"][0]
                validation_tokens = tokenizer_metadata[candidate["name"]][
                    "validation_source_exposure"
                ]["tokenizer_tokens"]
                validation_bytes = tokenizer_metadata[candidate["name"]][
                    "validation_source_exposure"
                ]["estimated_utf8_bytes"]
                rows.append(
                    {
                        "candidate": candidate["name"],
                        "seed": seed,
                        "learning_rate": learning_rate,
                        "tokenizer_fingerprint": fingerprint,
                        "initial_validation_loss": result["initial_validation_loss"],
                        "final_validation_loss": result["final_validation_loss"],
                        "validation_bits_per_estimated_source_byte": (
                            result["final_validation_loss"]
                            * validation_tokens
                            / max(validation_bytes, 1)
                            / math.log(2)
                        ),
                        "final_train_loss": result["final_train_loss"],
                        "consumed_tokens": result["consumed_tokens"],
                        "training_seconds": result["training_seconds"],
                        "tokens_per_second": result["tokens_per_second"],
                        "total_parameters": result["stats"]["total_parameters"],
                        "checkpoint": result["checkpoint"],
                        "arm_report": str(arm_report_path),
                    }
                )
    selections = {}
    for candidate in config["candidates"]:
        name = candidate["name"]
        candidate_rows = [row for row in rows if row["candidate"] == name]
        lr_means = {
            lr: statistics.fmean(
                row["final_validation_loss"]
                for row in candidate_rows
                if row["learning_rate"] == lr
            )
            for lr in config["learning_rates"]
        }
        selected_lr = min(lr_means, key=lambda lr: (lr_means[lr], lr))
        selections[name] = {
            "learning_rate": selected_lr,
            "mean_final_validation_loss": lr_means[selected_lr],
        }
    nominal_tokens = (
        config["optimizer_updates"]
        * config["gradient_accumulation_steps"]
        * config["batch_size"]
        * config["sequence_length"]
    )
    # A final partially filled packed block can make realized attention-mask
    # tokens slightly lower than the nominal rectangular batch count. The fair
    # gate is exact equality across arms plus a tight completeness threshold.
    matched = (
        max(row["consumed_tokens"] for row in rows)
        - min(row["consumed_tokens"] for row in rows)
        <= 0.001 * nominal_tokens
        and len({row["total_parameters"] for row in rows}) == 1
        and all(row["consumed_tokens"] >= 0.99 * nominal_tokens for row in rows)
    )
    report = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "passed": matched,
        "interpretation": config["interpretation"],
        "config_sha256": _sha256(config_path),
        "train_data_sha256": _sha256(Path(config["train_data"])),
        "validation_data_sha256": _sha256(Path(config["validation_data"])),
        "dataset_manifest_sha256": _sha256(Path(config["dataset_manifest"])),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": config["device"],
        },
        "tokenizers": tokenizer_metadata,
        "selections": selections,
        "results": rows,
        "training_seconds_sum": sum(row["training_seconds"] for row in rows),
        "orchestration_elapsed_seconds": time.perf_counter() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    report = run(args.config, args.report)
    print(
        json.dumps(
            {"passed": report["passed"], "selections": report["selections"]}, indent=2
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
