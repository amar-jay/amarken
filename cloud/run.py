"""Resumable single-GPU preparation -> baseline -> LoRA -> calibration -> reports."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(min_vram=24):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. The locked PyTorch build requires a compatible NVIDIA driver; setup does not modify host drivers."
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    free = shutil.disk_usage(ROOT / "runs" if (ROOT / "runs").exists() else ROOT).free
    if free < 50 * 1024**3:
        raise RuntimeError(
            "Keep at least 50 GiB free on the training volume for environments and checkpoints"
        )
    device = torch.cuda.get_device_properties(0)
    if device.total_memory < min_vram * 1024**3 * 0.95:
        raise RuntimeError(
            f"Job requires approximately {min_vram} GiB VRAM; lower the configured chunk size and pass a matching --min-vram for smaller GPUs"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "This cloud profile expects an Ampere-or-newer GPU with BF16 support"
        )
    (torch.ones(2, device="cuda", dtype=torch.bfloat16) * 2).sum().item()
    print(
        json.dumps(
            {
                "gpu": device.name,
                "vram_gib": device.total_memory / 1024**3,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="cloud/clinc.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--min-vram", type=float, default=24)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.preflight:
        preflight(args.min_vram)
        return
    config = read(args.config)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", config["name"]):
        raise ValueError("Job name must be a simple directory name")
    if (
        min(
            config["epochs"],
            config["batch_size"],
            config["candidate_chunk_size"],
            config["checkpoint_steps"],
        )
        < 1
    ):
        raise ValueError("Training dimensions must be positive")
    job = ROOT / "runs" / config["name"]
    prepared = ROOT / "data" / (config["name"] + "-prepared")
    outputs = ROOT / "outputs" / config["name"]
    prefix = [sys.executable, "-c", "from amarken import main; main()"]
    fallback = ["--fallback"] if config["fallback"] else []
    preparation = (
        prefix
        + [
            "prepare",
            "--dataset",
            config["dataset"],
            "--output",
            str(prepared),
            "--max-length",
            str(config["max_length"]),
        ]
        + fallback
    )
    if config.get("review_file"):
        preparation += ["--review", config["review_file"]]
    stages = [("prepare", preparation, prepared / "manifest.json")]
    for mode in (["head"] if config["include_head_baseline"] else []) + ["lora"]:
        run = job / mode
        command = (
            prefix
            + [
                "train",
                "--prepared",
                str(prepared),
                "--output",
                str(run),
                "--mode",
                mode,
                "--epochs",
                str(config["epochs"]),
                "--batch-size",
                str(config["batch_size"]),
                "--candidate-chunk-size",
                str(config["candidate_chunk_size"]),
                "--checkpoint-steps",
                str(config["checkpoint_steps"]),
            ]
            + fallback
        )
        if config.get("validation_steps"):
            command += ["--validation-steps", str(config["validation_steps"])]
        backup = config.get("checkpoint_backup", {})
        if backup.get("uri"):
            command += ["--backup-uri", backup["uri"].rstrip("/") + "/" + mode]
            if backup.get("endpoint_url"):
                command += ["--backup-endpoint-url", backup["endpoint_url"]]
        tracking = config.get("wandb", {})
        if tracking.get("mode", "disabled") != "disabled":
            command += [
                "--wandb-mode",
                tracking["mode"],
                "--wandb-project",
                tracking["project"],
                "--wandb-group",
                config["name"],
                "--wandb-name",
                f"{config['name']}-{mode}",
            ]
            if tracking.get("entity"):
                command += ["--wandb-entity", tracking["entity"]]
            if tracking.get("log_checkpoints", False):
                command += ["--wandb-checkpoints"]
        stages.append((f"train-{mode}", command, run / "best.json"))
        stages.append(
            (
                f"calibrate-{mode}",
                prefix
                + [
                    "calibrate",
                    "--model",
                    f"BEST:{mode}",
                    "--prepared",
                    str(prepared),
                    "--output",
                    str(outputs / f"{mode}-calibrated"),
                    "--device",
                    "cuda",
                    "--dtype",
                    "bfloat16",
                ],
                outputs / f"{mode}-calibrated" / "checksums.json",
            )
        )
        stages.append(
            (
                f"evaluate-{mode}",
                prefix
                + [
                    "evaluate",
                    "--model",
                    str(outputs / f"{mode}-calibrated"),
                    "--prepared",
                    str(prepared),
                    "--output",
                    str(outputs / f"{mode}-test"),
                    "--device",
                    "cuda",
                    "--dtype",
                    "bfloat16",
                    "--robustness",
                ],
                outputs / f"{mode}-test" / "report.json",
            )
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": config,
                    "stages": [{"name": n, "argv": a} for n, a, _ in stages],
                },
                indent=2,
            )
        )
        return
    preflight(args.min_vram)
    job.mkdir(parents=True, exist_ok=True)
    with (job / "job.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {
            "config": config,
            "source": {
                str(p.relative_to(ROOT)): digest(p)
                for p in [
                    ROOT / "uv.lock",
                    Path(__file__),
                    *sorted((ROOT / "src" / "amarken").glob("*.py")),
                ]
            },
        }
        identity_path = job / "job.json"
        if identity_path.exists() and read(identity_path) != identity:
            raise ValueError(
                "Job code/config differs; use the original bundle or a new job name"
            )
        write(identity_path, identity)
        status_path = job / "status.json"
        status = (
            read(status_path)
            if status_path.exists()
            else {"stages": {}, "pid": os.getpid()}
        )
        status["pid"] = os.getpid()
        environment = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": str(ROOT / ".cache" / "matplotlib"),
        }
        for name, command, marker in stages:
            previous = status["stages"].get(name, {})
            if previous.get("status") == "completed":
                if not marker.is_file() or digest(marker) != previous["fingerprint"]:
                    raise ValueError(f"Completed stage artifact changed: {name}")
                continue
            command = list(command)
            if not name.startswith("train-") and previous.get("status") in (
                "running",
                "failed",
            ):
                # Keep partial outputs for diagnosis and restart the stage cleanly.
                destination = marker.parent
                if destination.exists():
                    destination.rename(
                        destination.with_name(
                            destination.name + f".partial-{time.time_ns()}"
                        )
                    )
            if name.startswith("train-"):
                run = job / name.removeprefix("train-")
                if run.exists():
                    latest = read(run / "latest.json")
                    command += ["--resume", str(run / latest["checkpoint"])]
            command = [
                str(
                    job
                    / value.removeprefix("BEST:")
                    / read(job / value.removeprefix("BEST:") / "best.json")[
                        "checkpoint"
                    ]
                )
                if value.startswith("BEST:")
                else value
                for value in command
            ]
            status["stages"][name] = {
                "status": "running",
                "started": time.time(),
                "argv": command,
            }
            write(status_path, status)
            print(f"Starting {name}", flush=True)
            try:
                with (job / f"{name}.log").open("a") as stream:
                    subprocess.run(
                        command,
                        check=True,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                if not marker.is_file():
                    raise RuntimeError(f"Missing stage completion artifact: {marker}")
            except Exception as error:
                status["stages"][name].update(status="failed", error=str(error))
                write(status_path, status)
                raise
            status["stages"][name].update(
                status="completed", finished=time.time(), fingerprint=digest(marker)
            )
            write(status_path, status)
        status["status"] = "completed"
        write(status_path, status)
        print(
            f"Completed; results in {outputs}. Instance billing continues until stopped/deleted by its owner.",
            flush=True,
        )


if __name__ == "__main__":
    main()
