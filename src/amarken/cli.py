"""Command-line entry point for training and inference."""

import argparse
import json
from pathlib import Path

from amarken.inference import predict_example
from amarken.main import train_example
from amarken.trainer import TrainingConfig


def parser():
    root = argparse.ArgumentParser(description="amarken training boilerplate")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", type=Path, default=Path("configs/training-example.json")
    )
    commands = root.add_subparsers(dest="command", required=True)
    example = commands.add_parser(
        "example", parents=[common], help="Train the synthetic classification example"
    )
    example.add_argument("--resume", type=Path, help="Local checkpoint file to resume")
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
    source.add_argument("--input", help="JSON array: 8 features or a batch of samples")
    source.add_argument(
        "--input-file", type=Path, help="JSON file containing input features"
    )
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        config = TrainingConfig(**json.loads(args.config.read_text()))
        if args.command == "example":
            checkpoint_prefix, history = train_example(config, resume=args.resume)
            print(f"Checkpoint directory: checkpoints/{checkpoint_prefix}")
            for metrics in history:
                print(json.dumps(metrics))
        elif args.command == "infer":
            raw = (
                args.input_file.read_text()
                if args.input_file is not None
                else args.input
            )
            result = predict_example(
                args.checkpoint, json.loads(raw), device=config.device
            )
            print(json.dumps(result, allow_nan=False))
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        root.error(str(error))
