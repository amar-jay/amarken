"""Command-line entry point for the training boilerplate."""

import argparse


def parser():
    root = argparse.ArgumentParser(description="amarken training boilerplate")
    root.add_argument("command", choices=["example"])
    root.add_argument("--config", default="configs/training-example.json")
    root.add_argument("--resume")
    return root


def main():
    args = parser().parse_args()
    from amarken.example import main as example_main

    options = ["--config", args.config]
    if args.resume:
        options += ["--resume", args.resume]
    example_main(options)
