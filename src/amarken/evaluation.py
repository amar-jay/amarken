"""Calibration, scoring, baselines and local evaluation reports."""

import statistics
import time
from collections import defaultdict
from math import ceil, log
from pathlib import Path

import torch

from amarken.constants import DECISION_KINDS
from amarken.decision import decision_loss
from amarken.storage import (
    read_rows,
    verify_prepared,
    write_json,
)


def probabilities(logits, kind, temperature=1.0):
    tensor = torch.tensor(logits, dtype=torch.float64) / temperature
    if kind == "noul":
        p = tensor.sigmoid().item()
        return [1 - p, p]
    return tensor.softmax(-1).tolist()


def metrics(records, bins=10):
    if not records:
        return {"count": 0}
    correct, nll, brier = [], [], []
    classes = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    reliability = [[] for _ in range(bins)]
    ranking, ordinal, oos, binary = [], [], [], []
    for record in records:
        row, p = record["row"], record["probabilities"]
        target = (
            row["target"]
            if row["kind"] != "noul"
            else [1 - row["target"][0], row["target"][0]]
        )
        pred, gold = (
            max(range(len(p)), key=p.__getitem__),
            max(range(len(target)), key=target.__getitem__),
        )
        hit = int(pred == gold)
        correct.append(hit)
        nll.append(-sum(t * log(max(v, 1e-15)) for t, v in zip(target, p)))
        brier.append(
            (p[1] - target[1]) ** 2
            if row["kind"] == "noul"
            else sum((v - t) ** 2 for v, t in zip(p, target))
        )
        for index, option in enumerate(row["options"] or ["false", "true"]):
            key = (row["source"], row["question"], option)
            classes[key][0] += int(pred == index and gold == index)
            classes[key][1] += int(pred == index and gold != index)
            classes[key][2] += int(pred != index and gold == index)
        confidence = p[pred]
        # With soft supervision, target mass is the expected correctness of this choice.
        reliability[min(int(confidence * bins), bins - 1)].append(
            (confidence, target[pred])
        )
        ranking.append((confidence, 1 - target[pred]))
        if row["kind"] == "score":
            ordinal.append(
                abs(
                    sum(i * v for i, v in enumerate(p))
                    - sum(i * t for i, t in enumerate(target))
                )
            )
        if row["kind"] == "noul":
            binary.append((p[1], target[1] >= 0.5))
        if "oos" in row.get("options", []):
            index = row["options"].index("oos")
            oos.append((p[index], target[index] >= 0.5))
    per_class = {
        " | ".join(key): {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        }
        for key, (tp, fp, fn) in classes.items()
    }
    f1s = [value["f1"] for value in per_class.values() if value["f1"] is not None]
    diagram, ece = [], 0.0
    for i, bucket in enumerate(reliability):
        conf = statistics.mean(v[0] for v in bucket) if bucket else None
        accuracy = statistics.mean(v[1] for v in bucket) if bucket else None
        if bucket:
            ece += len(bucket) / len(records) * abs(conf - accuracy)
        diagram.append(
            {
                "lower": i / bins,
                "upper": (i + 1) / bins,
                "count": len(bucket),
                "confidence": conf,
                "expected_accuracy": accuracy,
            }
        )
    ranking.sort(reverse=True)
    coverage = []
    for share in (0.1, 0.25, 0.5, 0.75, 1.0):
        count = max(1, ceil(len(ranking) * share))
        threshold = ranking[count - 1][0]
        retained = [r for r in ranking if r[0] >= threshold]
        coverage.append(
            {
                "coverage": len(retained) / len(ranking),
                "threshold": threshold,
                "expected_error": statistics.mean(r[1] for r in retained),
                "count": len(retained),
            }
        )
    result = {
        "count": len(records),
        "accuracy": statistics.mean(correct),
        "macro_f1": statistics.mean(f1s),
        "nll": statistics.mean(nll),
        "brier": statistics.mean(brier),
        "ece": ece,
        "reliability": diagram,
        "per_class": per_class,
        "coverage_error": coverage,
        "ordinal_tier_mae": statistics.mean(ordinal) if ordinal else None,
        "metric_notes": "Accuracy/F1 use target argmax; ECE/coverage use target mass for soft labels. Binary Brier uses p_true; multiclass Brier sums classes. ECE uses 10 equal-width bins.",
    }
    mean = statistics.mean(correct)
    z = 1.96
    denominator = 1 + z * z / len(correct)
    center = (mean + z * z / (2 * len(correct))) / denominator
    margin = (
        z
        * ((mean * (1 - mean) + z * z / (4 * len(correct))) / len(correct)) ** 0.5
        / denominator
    )
    result["accuracy_wilson_95"] = [center - margin, center + margin]
    if binary:
        result["binary_at_0.5"] = {
            "precision": sum(p >= 0.5 and t for p, t in binary)
            / max(1, sum(p >= 0.5 for p, t in binary)),
            "recall": sum(p >= 0.5 and t for p, t in binary)
            / max(1, sum(t for _, t in binary)),
            "accuracy": statistics.mean((p >= 0.5) == t for p, t in binary),
        }
    if oos:
        positives, negatives = sum(t for _, t in oos), sum(not t for _, t in oos)
        rank_sum, start = 0.0, 0
        ordered = sorted(oos)
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and ordered[end][0] == ordered[start][0]:
                end += 1
            rank_sum += ((start + 1 + end) / 2) * sum(t for _, t in ordered[start:end])
            start = end
        result["oos"] = {
            "auroc": (
                (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
                if positives and negatives
                else None
            ),
            "accuracy_at_0.5": statistics.mean((p >= 0.5) == t for p, t in oos),
            "precision_at_0.5": sum(p >= 0.5 and t for p, t in oos)
            / max(1, sum(p >= 0.5 for p, t in oos)),
            "recall_at_0.5": sum(p >= 0.5 and t for p, t in oos) / max(1, positives),
        }
    return result


@torch.no_grad()
def collect(model, rows):
    model.eval()
    result, durations, model_durations = [], [], []
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for row in rows:
        sync()
        start = time.perf_counter()
        batches, _, inverse = model.candidate_chunks(row["state"], row)
        chunks, seconds = [], 0.0
        for batch in batches:
            tokens = model.base.tokenize(batch)
            sync()
            model_start = time.perf_counter()
            features = model.base(**tokens)
            chunks.append(model.head(features.to(model.head.weight.dtype)).squeeze(-1))
            sync()
            seconds += time.perf_counter() - model_start
        logits = torch.cat(chunks)[inverse].float().cpu().tolist()
        temperature = model.calibration[row["kind"]]["temperature"]
        result.append(
            {
                "row": row,
                "logits": logits,
                "probabilities": probabilities(logits, row["kind"], temperature),
            }
        )
        durations.append(time.perf_counter() - start)
        model_durations.append(seconds)
    return result, durations, model_durations


def fit_calibration(model, rows, artifact_id, manifest_id):
    records, _, _ = collect(model, rows)
    fitted = {}
    for kind in DECISION_KINDS:
        subset = [r for r in records if r["row"]["kind"] == kind]
        default = {"temperature": 1.0, "status": "not_fitted", "count": len(subset)}
        if len(subset) < 20:
            fitted[kind] = {**default, "reason": "fewer_than_20_examples"}
            continue
        parameter = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [parameter], max_iter=50, line_search_fn="strong_wolfe"
        )

        def loss_at(log_temperature, subset=subset):
            temperature = log_temperature.clamp(-5, 5).exp()
            return torch.stack(
                [
                    decision_loss(torch.tensor(r["logits"]) / temperature, r["row"])
                    for r in subset
                ]
            ).mean()

        def closure(optimizer=optimizer, parameter=parameter, loss_at=loss_at):
            optimizer.zero_grad()
            loss = loss_at(parameter)
            loss.backward()
            return loss

        before = float(loss_at(torch.zeros(())).detach())
        try:
            optimizer.step(closure)
            after = float(loss_at(parameter).detach())
            temperature = float(parameter.detach().clamp(-5, 5).exp())
            if not torch.isfinite(torch.tensor(after)) or after > before:
                raise ValueError("Calibration did not improve NLL")
            fitted[kind] = {
                "temperature": temperature,
                "status": "fitted",
                "count": len(subset),
                "nll_before": before,
                "nll_after": after,
            }
        except (ValueError, RuntimeError) as exc:
            fitted[kind] = {**default, "reason": str(exc)}
    fitted["provenance"] = {
        "artifact_id": artifact_id,
        "prepared_manifest": manifest_id,
        "split": "calibration",
    }
    return fitted


def baseline_records(rows, train_rows, majority=False):
    counts = defaultdict(lambda: defaultdict(float))
    if majority:
        for row in train_rows:
            options = row["options"] or ["false", "true"]
            target = (
                row["target"]
                if row["kind"] != "noul"
                else [1 - row["target"][0], row["target"][0]]
            )
            for option, mass in zip(options, target):
                counts[(row["source"], row["question"], row["kind"])][option] += mass
    records = []
    for row in rows:
        options = row["options"] or ["false", "true"]
        mapping = counts[(row["source"], row["question"], row["kind"])]
        p = (
            [mapping[o] + 1e-6 for o in options]
            if majority and mapping
            else [1.0] * len(options)
        )
        p = [v / sum(p) for v in p]
        records.append({"row": row, "probabilities": p})
    return records


def plot_reliability(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for name, summary in report.items():
        if (
            not isinstance(summary, dict)
            or not summary.get("count")
            or "reliability" not in summary
        ):
            continue
        points = [b for b in summary["reliability"] if b["count"]]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.plot(
            [p["confidence"] for p in points],
            [p["expected_accuracy"] for p in points],
            "o-",
        )
        ax.set(
            xlabel="Predicted probability",
            ylabel="Observed target mass",
            xlim=(0, 1),
            ylim=(0, 1),
            title=name,
        )
        fig.tight_layout()
        fig.savefig(Path(directory) / f"reliability-{name}.png", dpi=160)
        plt.close(fig)


def evaluate(
    model, prepared, output, split="test", limit=None, robustness=False, challenge=None
):
    manifest = verify_prepared(prepared)
    rows = read_rows(Path(prepared) / f"{split}.jsonl")
    if limit is not None:
        rows = rows[:limit]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not rows:
        raise ValueError(f"Empty evaluation split: {split}")
    start = time.perf_counter()
    # Warm-up is reported separately from the measured warmed loop.
    collect(model, rows[:1])
    cold = time.perf_counter() - start
    records, durations, model_durations = collect(model, rows)
    report = {"overall": metrics(records)}
    for kind in DECISION_KINDS:
        report[kind] = metrics([r for r in records if r["row"]["kind"] == kind])
    report["by_source"] = {
        source: metrics([r for r in records if r["row"]["source"] == source])
        for source in sorted({r["row"]["source"] for r in records})
    }
    train_rows = read_rows(Path(prepared) / "train.jsonl")
    report["uniform"] = metrics(baseline_records(rows, train_rows))
    report["majority"] = metrics(baseline_records(rows, train_rows, majority=True))
    ordered = sorted(durations)
    report["runtime"] = {
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.base.parameters()).dtype),
        "chunk_size": model.settings.candidate_chunk_size,
        "first_forward_seconds": cold,
        "p50_seconds": statistics.median(durations),
        "model_only_p50_seconds": statistics.median(model_durations),
        "model_only_p95_seconds": sorted(model_durations)[
            max(0, ceil(0.95 * len(model_durations)) - 1)
        ],
        "p95_seconds": ordered[max(0, ceil(0.95 * len(ordered)) - 1)],
        "questions_per_second": len(rows) / sum(durations),
        "peak_cuda_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "input_token_lengths": [r.get("max_prompt_tokens") for r in rows],
        "candidate_counts": [len(r["options"]) or 1 for r in rows],
        "timing_scope": "includes rendering, tokenization, transfer, scoring and probability normalization; batch=one question",
    }
    if robustness:
        deltas = []
        for row in rows[:100]:
            if row["kind"] == "noul":
                continue
            flipped = {**row, "options": list(reversed(row["options"]))}
            with torch.no_grad():
                original = model(row["state"], row)
                reverse = model(flipped["state"], flipped).flip(0)
            deltas.append(float((original - reverse).abs().max()))
        report["robustness"] = {
            "permutation_max_logit_delta": max(deltas, default=0),
            "count": len(deltas),
            "limitation": "Semantic ambiguity and injected-state labels require a reviewed challenge set; no automatic safety claim",
        }
    if challenge:
        from amarken.data import validate_decision

        challenge_rows = read_rows(challenge)
        if not challenge_rows:
            raise ValueError("Challenge file is empty")
        for row in challenge_rows:
            validate_decision(row)
        challenge_records, _, _ = collect(model, challenge_rows)
        report["challenge"] = {
            "overall": metrics(challenge_records),
            "by_type": {
                category: metrics(
                    [
                        r
                        for r in challenge_records
                        if r["row"].get("challenge_type", "unspecified") == category
                    ]
                )
                for category in sorted(
                    {row.get("challenge_type", "unspecified") for row in challenge_rows}
                )
            },
        }
    report["provenance"] = {
        "split": split,
        "limit": limit,
        "manifest": manifest["files"][split],
        "calibration": model.calibration,
    }
    write_json(output / "report.json", report)
    with (output / "predictions.jsonl").open("w") as stream:
        import json

        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    plot_reliability(report, output)
    return report
