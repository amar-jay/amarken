"""Evaluate proxy checkpoints on frozen EN/TR capability and systems gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import gc
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

# This must precede torch's CUDA initialization when deterministic GPU
# evaluation is requested; it is harmless for the default CPU worker.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sentencepiece as spm
import torch
import torch.nn.functional as F

from src.data.proxy import TOKEN, _canonical
from src.models import create_config, create_model
from src.training.data import PackedSequenceDataset
from src.training.proxy_experiment import _tokenize


CHOICES = ("A", "B", "C", "D")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_choice(model, prompt_ids: list[int], choice_ids: list[int], max_length: int) -> float:
    # Preserve the complete candidate and crop only excess prompt prefix. Every
    # benchmark is authored to fit, while this guard prevents accidental overflow.
    prompt_ids = prompt_ids[-(max_length - len(choice_ids)):]
    tokens = torch.tensor([prompt_ids + choice_ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.inference_mode():
        logits = model(tokens).logits[0]
        log_probs = F.log_softmax(logits.float(), dim=-1)
    start = len(prompt_ids) - 1
    return sum(float(log_probs[start + offset, token]) for offset, token in enumerate(choice_ids))


def _capabilities(model, processor, benchmark: dict) -> tuple[list[dict], dict]:
    details = []
    for task in benchmark["tasks"]:
        suffix = "\nYanıt:" if task["language"] == "tr" else "\nAnswer:"
        prompt_ids = processor.encode(task["prompt"] + suffix, out_type=int)
        scores = [_score_choice(model, prompt_ids, processor.encode(" " + choice, out_type=int), model.config.max_position_embeddings) for choice in CHOICES]
        probabilities = torch.softmax(torch.tensor(scores, dtype=torch.float64), dim=0).tolist()
        prediction_index = max(range(len(CHOICES)), key=lambda index: (probabilities[index], -index))
        target_index = CHOICES.index(task["target"])
        details.append({
            "id": task["id"], "language": task["language"], "category": task["category"],
            "target": task["target"], "prediction": CHOICES[prediction_index],
            "correct": prediction_index == target_index,
            "confidence": probabilities[prediction_index],
            "target_probability": probabilities[target_index],
            "choice_probabilities": dict(zip(CHOICES, probabilities)),
        })
    grouped = defaultdict(list)
    for detail in details:
        grouped[(detail["category"], detail["language"])].append(detail)
    metrics = {
        f"{category}:{language}": {
            "count": len(values),
            "accuracy": sum(value["correct"] for value in values) / len(values),
            "mean_target_probability": statistics.fmean(value["target_probability"] for value in values),
        }
        for (category, language), values in sorted(grouped.items())
    }
    for language in ("en", "tr"):
        values = [detail for detail in details if detail["language"] == language]
        metrics[f"overall:{language}"] = {"count": len(values), "accuracy": sum(value["correct"] for value in values) / len(values)}
    metrics["overall"] = {"count": len(details), "accuracy": sum(value["correct"] for value in details) / len(details)}
    # Multiclass Brier/NLL measure probability quality, while five-bin ECE tests
    # whether stated choice confidence matches empirical correctness frequency.
    brier = statistics.fmean(
        sum((probability - (index == CHOICES.index(detail["target"]))) ** 2 for index, probability in enumerate(detail["choice_probabilities"].values()))
        for detail in details
    )
    nll = statistics.fmean(-math.log(max(detail["target_probability"], 1e-300)) for detail in details)
    ece = 0.0
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        values = [detail for detail in details if lower <= detail["confidence"] < lower + 0.2 or (lower == 0.8 and detail["confidence"] == 1.0)]
        if values:
            accuracy = sum(value["correct"] for value in values) / len(values)
            confidence = statistics.fmean(value["confidence"] for value in values)
            ece += len(values) / len(details) * abs(accuracy - confidence)
    metrics["calibration"] = {"multiclass_brier": brier, "choice_nll": nll, "ece_5_bin": ece}
    return details, metrics


@torch.no_grad()
def _validation_loss(model, dataset: PackedSequenceDataset) -> float:
    weighted = targets = 0
    model.eval()
    for index in range(len(dataset)):
        batch = dataset.batch([index], next(model.parameters()).device)
        count = int((batch["labels"][:, 1:] != -100).sum())
        if not count:
            continue
        loss = model(**batch).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite validation loss")
        weighted += float(loss) * count
        targets += count
    return weighted / targets


def _current_rss_bytes() -> int:
    return int(Path("/proc/self/statm").read_text().split()[1]) * resource.getpagesize()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _systems(model, dataset: PackedSequenceDataset, tokenizer_bytes: int, baseline_rss_bytes: int) -> dict:
    model.eval()
    device = next(model.parameters()).device
    batch = dataset.batch([0], device)
    for _ in range(3):
        with torch.inference_mode():
            model(batch["input_ids"], attention_mask=batch["attention_mask"])
    forward_times = []
    for _ in range(15):
        _synchronize(device)
        started = time.perf_counter_ns()
        with torch.inference_mode():
            model(batch["input_ids"], attention_mask=batch["attention_mask"])
        _synchronize(device)
        forward_times.append((time.perf_counter_ns() - started) / 1e6)
    generation_times = []
    prompt = batch["input_ids"][:, :16]
    for _ in range(5):
        _synchronize(device)
        started = time.perf_counter_ns()
        model.generate(prompt, max_new_tokens=8, temperature=0.0)
        _synchronize(device)
        generation_times.append((time.perf_counter_ns() - started) / 1e6)
    ordered = sorted(forward_times)
    stats = model.stats(dataset.sequence_length, element_bytes=2)
    # Linux statm reports current resident pages, unlike ru_maxrss which retains
    # the checkpoint deserialization peak. Recording both separates serving RAM
    # after optimizer-state release from the cost of loading a trainer checkpoint.
    resident_bytes = _current_rss_bytes()
    return {
        "forward_batch1_context_ms_median": statistics.median(forward_times),
        "forward_batch1_context_ms_p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "autoregressive_8_tokens_ms_median": statistics.median(generation_times),
        "autoregressive_tokens_per_second": 8_000 / statistics.median(generation_times),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "resident_process_rss_bytes": resident_bytes,
        "resident_rss_delta_from_framework_bytes": max(0, resident_bytes - baseline_rss_bytes),
        "kv_cache_bytes_context": stats.kv_cache_bytes,
        "deploy_model_bytes": stats.artifact_bytes,
        "tokenizer_bytes": tokenizer_bytes,
        "deploy_model_plus_tokenizer_bytes": stats.artifact_bytes + tokenizer_bytes,
        "training_parameter_bytes": stats.training_parameter_bytes,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }


def _worker(checkpoint: Path, tokenizer_path: Path, benchmark_path: Path, validation_path: Path, output: Path, device_name: str = "cpu") -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Baseline includes imported Python/PyTorch/SentencePiece libraries but no
    # tokenizer, model, checkpoint tensors, validation blocks, or benchmark.
    baseline_rss_bytes = _current_rss_bytes()
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model_type = payload["model_type"]
    # Copy scalar identity before deleting the potentially optimizer-heavy payload
    # so resident inference memory excludes checkpoint-only objects.
    variant = payload.get("metadata", {}).get("variant", model_type)
    model = create_model(create_config(model_type, **payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    # Trainer checkpoints include Adam moments; release them before measuring the
    # inference resident set. ru_maxrss separately discloses their load-time peak.
    del payload
    gc.collect()
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    validation = PackedSequenceDataset(_tokenize(validation_path, processor, 8_192), 64, processor.eos_id(), processor.pad_id())
    details, capability = _capabilities(model, processor, benchmark)
    tokenizer_bytes = tokenizer_path.stat().st_size + tokenizer_path.with_suffix(".vocab").stat().st_size
    result = {
        "model_type": model_type, "checkpoint": str(checkpoint),
        "variant": variant,
        "checkpoint_sha256": _sha256(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "validation_loss_full": _validation_loss(model, validation),
        "capability": capability, "task_details": details,
        "systems": _systems(model, validation, tokenizer_bytes, baseline_rss_bytes),
    }
    output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")


def _contamination_scan(train_path: Path, benchmark_path: Path, n: int = 13) -> dict:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    hashes = set()
    for task in benchmark["tasks"]:
        tokens = TOKEN.findall(_canonical(task["prompt"]))
        for index in range(len(tokens) - n + 1):
            raw = "\x1f".join(tokens[index:index + n]).encode()
            hashes.add(hashlib.blake2b(raw, digest_size=8).digest())
    matches = []
    with train_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            tokens = TOKEN.findall(_canonical(row["text"]))
            contaminated = False
            for index in range(len(tokens) - n + 1):
                raw = "\x1f".join(tokens[index:index + n]).encode()
                if hashlib.blake2b(raw, digest_size=8).digest() in hashes:
                    contaminated = True
                    break
            if contaminated:
                matches.append({"line": line_number, "document_id": row["id"]})
    return {"ngram_tokens": n, "matching_documents": len(matches), "matches": matches}


def run(proxy_report_path: Path, benchmark_path: Path, report_path: Path, device: str = "cpu") -> dict:
    proxy = json.loads(proxy_report_path.read_text(encoding="utf-8"))
    tokenizer = Path(proxy["shared"]["tokenizer"])
    train_data = Path(proxy["shared"]["train_data"])
    validation_data = Path(proxy["shared"]["validation_data"])
    contamination = _contamination_scan(train_data, benchmark_path)
    results = []
    with tempfile.TemporaryDirectory(prefix="amarken-eval-") as directory:
        for candidate in proxy["results"]:
            # Scaling reports intentionally retain inference weights for only one
            # seed; rows without a checkpoint are training statistics, not evaluable.
            if not candidate.get("checkpoint"):
                continue
            variant = candidate.get("variant", candidate["model_type"])
            output = Path(directory) / f"{variant}-seed-{candidate.get('seed', 'single')}.json"
            command = [
                sys.executable, "-m", "src.evaluation.capability", "--worker",
                "--checkpoint", candidate["checkpoint"]["path"], "--tokenizer", str(tokenizer),
                "--benchmark", str(benchmark_path), "--validation", str(validation_data), "--worker-output", str(output),
                "--device", device,
            ]
            subprocess.run(command, check=True)
            results.append(json.loads(output.read_text(encoding="utf-8")))
    report = {
        "format_version": 1, "benchmark_id": json.loads(benchmark_path.read_text())["benchmark_id"],
        "passed": contamination["matching_documents"] == 0 and all(result["capability"]["overall"]["count"] == 30 for result in results),
        "interpretation": "deterministic proxy diagnostics; not a capability claim",
        "benchmark_sha256": _sha256(benchmark_path), "proxy_experiment_sha256": _sha256(proxy_report_path),
        "train_data_sha256": _sha256(train_data), "validation_data_sha256": _sha256(validation_data),
        "tokenizer_sha256": _sha256(tokenizer), "contamination": contamination,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "sentencepiece": spm.__version__, "device": device, "torch_threads": 1},
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-report", type=Path, default=Path("experiments/proxy_10m.json"))
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/proxy_capability_v1.json"))
    parser.add_argument("--report", type=Path, default=Path("experiments/capability_10m.json"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.worker:
        _worker(args.checkpoint, args.tokenizer, args.benchmark, args.validation, args.worker_output, args.device)
        return 0
    report = run(args.proxy_report, args.benchmark, args.report, args.device)
    for result in report["results"]:
        print(result["model_type"], result["validation_loss_full"], result["capability"]["overall"]["accuracy"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
