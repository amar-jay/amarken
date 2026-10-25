"""Local preparation, training, calibration and prediction commands."""

import argparse
import json
import shutil
from pathlib import Path

from amarken.constants import (
    BACKUP_SELECTORS,
    BASES,
    DATASETS,
    EVAL_SPLITS,
    TRAIN_MODES,
    WANDB_MODES,
    WANDB_PROJECT,
)
from amarken.storage import (
    fingerprint,
    read_json,
    read_rows,
    verify_prepared,
    write_json,
)


def parser():
    root = argparse.ArgumentParser(description="Amarken decision model pipeline")
    commands = root.add_subparsers(dest="command")
    smoke = commands.add_parser("smoke", help="Load the pretrained feature model")
    for command in (root, smoke):
        command.add_argument("--fallback", action="store_true")
        command.add_argument("--device", default="cpu")
        command.add_argument(
            "--dtype", choices=("float32", "bfloat16", "float16"), default="float32"
        )
        command.add_argument("--local-files-only", action="store_true")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset", choices=DATASETS, default=DATASETS[0])
    prepare.add_argument("--output", required=True)
    prepare.add_argument(
        "--review",
        help="JSON with reviewed_sources and optional source_families mapping",
    )
    prepare.add_argument("--fallback", action="store_true")
    prepare.add_argument("--max-length", type=int, default=2048)
    prepare.add_argument("--max-examples", type=int, default=100000)
    prepare.add_argument("--max-per-source", type=int, default=2000)
    prepare.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train")
    train.add_argument("--prepared", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--mode", choices=TRAIN_MODES, default=TRAIN_MODES[0])
    train.add_argument("--fallback", action="store_true")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--candidate-chunk-size", type=int, default=1)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--checkpoint-steps", type=int, default=100)
    train.add_argument("--validation-steps", type=int)
    train.add_argument("--backup-uri", help="s3://bucket/unique-run-prefix")
    train.add_argument("--backup-endpoint-url")
    train.add_argument("--cpu-smoke", action="store_true")
    train.add_argument(
        "--smoke",
        action="store_true",
        help="Bounded GPU smoke run, not quality evaluation",
    )
    train.add_argument("--resume")
    train.add_argument("--wandb-mode", choices=WANDB_MODES, default=WANDB_MODES[0])
    train.add_argument("--wandb-project", default=WANDB_PROJECT)
    train.add_argument("--wandb-entity")
    train.add_argument("--wandb-name")
    train.add_argument("--wandb-group")
    train.add_argument("--wandb-checkpoints", action="store_true")
    for name in ("calibrate", "evaluate", "predict"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument(
            "--dtype", choices=("float32", "bfloat16"), default="float32"
        )
        if name == "predict":
            command.add_argument(
                "--input", required=True, help="JSON file with state and questions"
            )
        else:
            command.add_argument("--prepared", required=True)
            command.add_argument("--output", required=True)
        if name == "evaluate":
            command.add_argument("--split", choices=EVAL_SPLITS, default="test")
            command.add_argument("--limit", type=int)
            command.add_argument("--robustness", action="store_true")
            command.add_argument("--challenge", help="Reviewed labeled challenge JSONL")
    restore = commands.add_parser("restore")
    restore.add_argument("--backup-uri", required=True)
    restore.add_argument("--backup-endpoint-url")
    restore.add_argument(
        "--which", choices=BACKUP_SELECTORS, default=BACKUP_SELECTORS[0]
    )
    restore.add_argument("--output", required=True)
    restore.add_argument("--prepared-output", required=True)
    return root


def main():
    args = parser().parse_args()
    from amarken.base import BaseConfig, BaseModel

    if args.command in (None, "smoke"):
        import torch

        config = BaseConfig(
            model_id=BASES[int(args.fallback)], device=args.device, dtype=args.dtype
        )
        model = BaseModel.load(config, local_files_only=args.local_files_only)
        with torch.inference_mode():
            features = model(
                **model.tokenize(["State: payment failed. Question: Is action needed?"])
            )
        if not torch.isfinite(features).all():
            raise RuntimeError("Non-finite features")
        print(
            json.dumps(
                {
                    "model": config.model_id,
                    "revision": config.revision,
                    "feature_shape": list(features.shape),
                }
            )
        )
    elif args.command == "prepare":
        from transformers import AutoTokenizer

        from amarken.data import DataConfig
        from amarken.prepare import PrepareConfig, prepare

        config = BaseConfig(
            model_id=BASES[int(args.fallback)], max_length=args.max_length
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=False,
            cache_dir=config.cache_dir,
        )
        review = read_json(args.review) if args.review else {}
        data = DataConfig(
            dataset_id=args.dataset,
            seed=args.seed,
            max_train_examples=args.max_examples,
            max_per_source=args.max_per_source,
        )
        settings = PrepareConfig(
            max_length=args.max_length,
            reviewed_sources=tuple(review.get("reviewed_sources", [])),
            source_families=tuple(review.get("source_families", {}).items()),
        )
        manifest = prepare(
            args.output,
            tokenizer,
            data,
            settings,
            tokenizer_metadata={
                "model_id": config.model_id,
                "revision": config.revision,
            },
        )
        print(json.dumps(manifest["counts"]))
    elif args.command == "restore":
        from amarken.backup import BackupConfig, CheckpointBackup

        result = CheckpointBackup(
            BackupConfig(args.backup_uri, args.backup_endpoint_url)
        ).restore(args.output, args.prepared_output, which=args.which)
        print(json.dumps(result, indent=2))
    elif args.command == "train":
        import torch

        from amarken.backup import BackupConfig
        from amarken.decision import DecisionConfig
        from amarken.tracking import WandbConfig
        from amarken.training import TrainConfig, train

        device = "cpu" if args.cpu_smoke else "cuda"
        dtype = (
            "bfloat16"
            if device == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            else "float32"
        )
        base = BaseConfig(
            model_id=BASES[int(args.fallback)],
            device=device,
            dtype=dtype,
            max_length=verify_prepared(args.prepared)["prepare_config"]["max_length"],
        )
        config = TrainConfig(
            mode=args.mode,
            epochs=args.epochs,
            effective_batch_size=args.batch_size,
            max_steps=args.max_steps,
            checkpoint_steps=args.checkpoint_steps,
            validation_steps=args.validation_steps,
            cpu_smoke=args.cpu_smoke,
            smoke=args.smoke,
        )
        print(
            train(
                args.prepared,
                args.output,
                config,
                base,
                DecisionConfig(candidate_chunk_size=args.candidate_chunk_size),
                resume=args.resume,
                backup=BackupConfig(args.backup_uri, args.backup_endpoint_url),
                tracking=WandbConfig(
                    mode=args.wandb_mode,
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_name,
                    group=args.wandb_group,
                    log_checkpoints=args.wandb_checkpoints,
                ),
            )
        )
    else:
        from amarken.decision import DecisionModel

        model = DecisionModel.load(args.model, device=args.device, dtype=args.dtype)
        if args.command == "predict":
            request = read_json(args.input)
            print(
                json.dumps(
                    model.predict(request["state"], request["questions"]), indent=2
                )
            )
        elif args.command == "calibrate":
            from amarken.evaluation import fit_calibration

            verify_prepared(args.prepared)
            prepared_id = fingerprint(Path(args.prepared) / "manifest.json")
            provenance = model.training_provenance or {}
            if provenance.get("prepared_id") != prepared_id:
                raise ValueError(
                    "Calibration must use the prepared partitions recorded during training"
                )
            rows = read_rows(Path(args.prepared) / "calibration.jsonl")
            if not rows:
                raise ValueError("Calibration data is empty")
            if Path(args.output).exists():
                raise FileExistsError(args.output)
            calibration = fit_calibration(
                model,
                rows,
                fingerprint(Path(args.model) / "checksums.json"),
                fingerprint(Path(args.prepared) / "manifest.json"),
            )
            shutil.copytree(args.model, args.output)
            output = Path(args.output)
            write_json(output / "calibration.json", calibration)
            write_json(
                output / "checksums.json",
                {
                    str(p.relative_to(output)): fingerprint(p)
                    for p in output.rglob("*")
                    if p.is_file() and p.name != "checksums.json"
                },
            )
            print(json.dumps(calibration, indent=2))
        elif args.command == "evaluate":
            from amarken.evaluation import evaluate

            if args.limit is not None and args.limit < 1:
                raise ValueError("limit must be positive")
            report = evaluate(
                model,
                args.prepared,
                args.output,
                args.split,
                args.limit,
                args.robustness,
                challenge=args.challenge,
            )
            print(
                json.dumps(
                    {
                        key: report["overall"][key]
                        for key in (
                            "count",
                            "accuracy",
                            "macro_f1",
                            "nll",
                            "brier",
                            "ece",
                        )
                    },
                    indent=2,
                )
            )
