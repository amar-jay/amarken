"""Command-line entry point for training and inference."""

import argparse
import json
from pathlib import Path

from amarken import distributed
from amarken.inference import run_inference
from amarken.main import train_example
from amarken.trainer import TrainingConfig
from amarken.validation import run_validation


def parser():
    root = argparse.ArgumentParser(description="amarken training boilerplate")
    common = argparse.ArgumentParser(add_help=False)
    if len(list(Path("configs").glob("*.json"))) == 1:
        common.add_argument(
            "--config",
            type=Path,
            default=next(Path("configs").glob("*.json")),
        )
    else:
        common.add_argument(
            "--config", type=Path, required=True, help="JSON config file"
        )
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train", parents=[common], help="Train the selected model and dataset"
    )
    train.add_argument("--resume", type=Path, help="Local checkpoint file to resume")
    infer = commands.add_parser(
        "infer", parents=[common], help="Run inference from a saved checkpoint"
    )
    infer.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Local best.pt or latest.pt training checkpoint",
    )
    source = infer.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", help="JSON input accepted by the selected model implementation"
    )
    source.add_argument(
        "--input-file", type=Path, help="JSON file containing input features"
    )
    validate = commands.add_parser(
        "validate",
        parents=[common],
        help="Evaluate a checkpoint on the validation split",
    )
    validate.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Local best.pt or latest.pt checkpoint",
    )
    test = commands.add_parser(
        "test",
        parents=[common],
        help="Evaluate a checkpoint on the held-out test split",
    )
    test.add_argument("--checkpoint", type=Path, required=True)
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command != "train" and distributed.launched():
            raise ValueError(
                "Only train supports torchrun; run evaluation and inference in one process"
            )
        config = TrainingConfig(**json.loads(args.config.read_text()))
        if args.command == "train":
            checkpoint_prefix, history = train_example(config, resume=args.resume)
            if distributed.primary():
                print(f"Checkpoint directory: checkpoints/{checkpoint_prefix}")
                for metrics in history:
                    print(json.dumps(metrics))
        elif args.command == "test":
            print(
                json.dumps(
                    run_validation(args.checkpoint, config, split="test"),
                    allow_nan=False,
                )
            )
        elif args.command == "validate":
            print(json.dumps(run_validation(args.checkpoint, config), allow_nan=False))
        elif args.command == "infer":
            raw = (
                args.input_file.read_text()
                if args.input_file is not None
                else args.input
            )
            result = run_inference(
                args.checkpoint,
                json.loads(raw),
                device=config.device,
                precision=config.precision,
            )
            print(json.dumps(result, allow_nan=False))
    except (OSError, ValueError, TypeError, RuntimeError, NotImplementedError) as error:
        root.error(str(error))
