"""Evaluate a checkpoint on the meaningful-scale v2 benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import unicodedata

import torch
import torch.nn.functional as F

from src.models import create_config, create_model
from src.tokenization import load_tokenizer, tokenizer_fingerprint

LETTERS = "ABCD"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wilson(correct: int, count: int, z: float = 1.959963984540054) -> dict:
    if count == 0:
        return {"accuracy": None, "low": None, "high": None, "count": 0}
    p = correct / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    margin = (
        z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    )
    return {
        "accuracy": p,
        "low": center - margin,
        "high": center + margin,
        "count": count,
    }


def _normalize_exact(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).strip().split())


def _score_completion(
    model, prefix: list[int], completion: list[int]
) -> tuple[float, int]:
    if not completion:
        raise ValueError("completion encodes to zero tokens")
    maximum = model.config.max_position_embeddings
    prefix = prefix[-max(1, maximum - len(completion)) :]
    ids = prefix + completion
    tensor = torch.tensor(
        [ids], dtype=torch.long, device=next(model.parameters()).device
    )
    with torch.inference_mode():
        log_probs = F.log_softmax(model(tensor).logits[0].float(), dim=-1)
    start = len(prefix) - 1
    return sum(
        float(log_probs[start + offset, token])
        for offset, token in enumerate(completion)
    ), len(completion)


def _render_mc(task: dict, options: list[str]) -> str:
    choices = " ".join(
        f"{letter}) {option}" for letter, option in zip(LETTERS, options)
    )
    suffix = "Yanıt:" if task["language"] == "tr" else "Answer:"
    return f"{task['prompt']}\n{choices}\n{suffix}"


def _multiple_choice(model, tokenizer, benchmark: dict) -> tuple[list[dict], dict]:
    details = []
    for task in benchmark["multiple_choice"]:
        predictions = []
        for shift in benchmark["permutations"]:
            options = task["options"][shift:] + task["options"][:shift]
            target_index = options.index(task["answer"])
            prompt_ids = tokenizer.encode(_render_mc(task, options))
            scores = [
                _score_completion(model, prompt_ids, tokenizer.encode(" " + letter))[0]
                for letter in LETTERS
            ]
            probabilities = torch.softmax(
                torch.tensor(scores, dtype=torch.float64), dim=0
            ).tolist()
            prediction_index = max(
                range(4), key=lambda index: (probabilities[index], -index)
            )
            predictions.append(options[prediction_index])
            details.append(
                {
                    "id": task["id"],
                    "language": task["language"],
                    "category": task["category"],
                    "permutation": shift,
                    "target_position": target_index,
                    "prediction_position": prediction_index,
                    "prediction": options[prediction_index],
                    "answer": task["answer"],
                    "correct": prediction_index == target_index,
                    "target_probability": probabilities[target_index],
                    "probabilities": probabilities,
                }
            )
        for row in details[-len(benchmark["permutations"]) :]:
            row["content_invariant_across_permutations"] = len(set(predictions)) == 1
    metrics = _group_accuracy(details)
    task_groups = defaultdict(list)
    for row in details:
        task_groups[row["id"]].append(row)
    metrics["permutation"] = {
        "tasks": len(task_groups),
        "all_permutations_correct": sum(
            all(row["correct"] for row in rows) for rows in task_groups.values()
        )
        / len(task_groups),
        "content_invariant": sum(
            len({row["prediction"] for row in rows}) == 1
            for rows in task_groups.values()
        )
        / len(task_groups),
    }
    return details, metrics


def _group_accuracy(details: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in details:
        groups[(row["language"], row["category"])].append(row)
        groups[(row["language"], "overall")].append(row)
    groups[("all", "overall")] = details
    return {
        f"{language}:{category}": _wilson(
            sum(row["correct"] for row in rows), len(rows)
        )
        for (language, category), rows in sorted(groups.items())
    }


def _generative(model, tokenizer, benchmark: dict) -> tuple[list[dict], dict]:
    details = []
    for task in benchmark["generative"]:
        prompt = tokenizer.encode(task["prompt"])
        prompt = prompt[-model.config.max_position_embeddings :]
        tensor = torch.tensor(
            [prompt], dtype=torch.long, device=next(model.parameters()).device
        )
        with torch.inference_mode():
            output = model.generate(
                tensor,
                max_new_tokens=task["max_new_tokens"],
                temperature=0.0,
                eos_token_id=tokenizer.eos_id(),
            )
        generated = tokenizer.decode(output[0, len(prompt) :].tolist())
        normalized = _normalize_exact(generated)
        accepted = {_normalize_exact(answer) for answer in task["answers"]}
        details.append(
            {
                "id": task["id"],
                "language": task["language"],
                "category": task["category"],
                "generated": generated,
                "normalized": normalized,
                "answers": task["answers"],
                "correct": normalized in accepted,
            }
        )
    return details, _group_accuracy(details)


def _language_model(model, tokenizer, benchmark: dict) -> tuple[list[dict], dict]:
    details = []
    for task in benchmark["language_model"]:
        score, tokens = _score_completion(
            model,
            tokenizer.encode(task["prefix"]),
            tokenizer.encode(task["continuation"]),
        )
        details.append(
            {
                "id": task["id"],
                "language": task["language"],
                "category": task["category"],
                "document_id": task["document_id"],
                "tokens": tokens,
                "nll": -score,
            }
        )
    groups = defaultdict(list)
    for row in details:
        groups[row["language"]].append(row)
        groups["all"].append(row)
    metrics = {}
    for language, rows in groups.items():
        tokens = sum(row["tokens"] for row in rows)
        nll = sum(row["nll"] for row in rows)
        metrics[language] = {
            "documents": len(rows),
            "tokens": tokens,
            "nll_per_token": nll / tokens,
            "perplexity": math.exp(min(20, nll / tokens)),
        }
    return details, metrics


def run(
    checkpoint: Path,
    tokenizer_path: Path,
    benchmark_path: Path,
    report_path: Path,
    device_name: str = "cpu",
) -> dict:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but unavailable")
    torch.use_deterministic_algorithms(True)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    tokenizer = load_tokenizer(tokenizer_path)
    expected = payload.get("metadata", {}).get("tokenizer_fingerprint")
    if expected and expected != tokenizer_fingerprint(tokenizer):
        raise ValueError("checkpoint tokenizer fingerprint mismatch")
    model = create_model(
        create_config(payload["model_type"], **payload["model_config"])
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    mc_details, mc_metrics = _multiple_choice(model, tokenizer, benchmark)
    gen_details, gen_metrics = _generative(model, tokenizer, benchmark)
    lm_details, lm_metrics = _language_model(model, tokenizer, benchmark)
    report = {
        "format_version": 2,
        "benchmark_id": benchmark["benchmark_id"],
        "passed": True,
        "interpretation": "deterministic development diagnostics; secret holdout still required for final claims",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "tokenizer": str(tokenizer_path),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "benchmark_sha256": _sha256(benchmark_path),
        "multiple_choice": {"metrics": mc_metrics, "details": mc_details},
        "generative": {"metrics": gen_metrics, "details": gen_details},
        "language_model": {"metrics": lm_metrics, "details": lm_details},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizers/v2/tiktoken-tr-bpe-12k.json"),
    )
    parser.add_argument(
        "--benchmark", type=Path, default=Path("benchmarks/meaningful_scale_v2.json")
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run(
        args.checkpoint, args.tokenizer, args.benchmark, args.report, args.device
    )
    print(
        json.dumps(
            {
                "multiple_choice": result["multiple_choice"]["metrics"]["all:overall"],
                "generative": result["generative"]["metrics"]["all:overall"],
                "language_model": result["language_model"]["metrics"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
