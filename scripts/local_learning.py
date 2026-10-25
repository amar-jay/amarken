"""Bounded local CLINC learning experiment; never consumes the test set."""

import argparse
import random
import time
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import torch

from amarken.base import BaseConfig, BaseModel
from amarken.constants import DATA_SPLITS
from amarken.decision import DecisionConfig, DecisionModel
from amarken.evaluation import evaluate
from amarken.storage import (
    fingerprint,
    read_json,
    read_rows,
    verify_prepared,
    write_json,
)
from amarken.tracking import WandbConfig
from amarken.training import TrainConfig, train

ROOT = Path(__file__).resolve().parents[1]


def label(row):
    return row["options"][max(range(len(row["target"])), key=row["target"].__getitem__)]


def prepare_subset(config, directory):
    source = ROOT / config["prepared_source"]
    manifest = verify_prepared(source)
    pools = {}
    for split in ("train", "validation"):
        groups = defaultdict(list)
        for row in read_rows(source / f"{split}.jsonl"):
            groups[label(row)].append(row)
        pools[split] = groups
    eligible = [
        intent
        for intent in pools["train"]
        if len(pools["train"][intent]) >= config["train_per_intent"]
        and len(pools["validation"][intent]) >= config["validation_per_intent"]
    ]
    eligible.sort(
        key=lambda intent: sha256(f"{config['seed']}:{intent}".encode()).hexdigest()
    )
    if "oos" not in eligible:
        raise ValueError("Expected eligible out-of-scope examples")
    intents = ["oos"] + [i for i in eligible if i != "oos"][
        : config["intent_count"] - 1
    ]
    if len(intents) != config["intent_count"]:
        raise ValueError("Insufficient eligible intents")
    selected = {}
    for split, count in (
        ("train", config["train_per_intent"]),
        ("validation", config["validation_per_intent"]),
    ):
        rows = []
        for intent in intents:
            candidates = sorted(
                pools[split][intent],
                key=lambda row: sha256(
                    f"{config['seed']}:{row['id']}".encode()
                ).hexdigest(),
            )
            rows.extend(candidates[:count])
        random.Random(config["seed"]).shuffle(rows)
        selected[split] = rows
    train_groups = {row["group_id"] for row in selected["train"]}
    if train_groups & {row["group_id"] for row in selected["validation"]}:
        raise ValueError("Subset contains overlapping groups")
    directory.mkdir(parents=True, exist_ok=False)
    import json

    for split in DATA_SPLITS:
        with (directory / f"{split}.jsonl").open("w") as stream:
            for row in selected.get(split, []):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (directory / "exclusions.jsonl").write_text("")
    subset_counts = Counter(
        f"{split}:{row['source']}:{row['kind']}"
        for split, rows in selected.items()
        for row in rows
    )
    manifest.update(
        source_kind_counts=dict(subset_counts),
        exclusions={},
        exclusion_fingerprint=fingerprint(directory / "exclusions.jsonl"),
        counts={split: len(selected.get(split, [])) for split in DATA_SPLITS},
        files={
            split: fingerprint(directory / f"{split}.jsonl") for split in DATA_SPLITS
        },
        subset_protocol={
            "config": config,
            "script_sha256": fingerprint(__file__),
            "intents": intents,
            "parent_manifest_sha256": fingerprint(source / "manifest.json"),
            "purpose": "learning check; all 151 candidates retained; calibration/test not sampled",
        },
    )
    write_json(directory / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local-learning.json")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA is unavailable")
    prepared = ROOT / "data" / config["name"]
    output = ROOT / "runs" / config["name"]
    reports = ROOT / "outputs" / config["name"]
    reports.mkdir(parents=True, exist_ok=False)
    manifest = prepare_subset(config, prepared)
    write_json(reports / "protocol.json", config)
    write_json(reports / "status.json", {"stage": "baseline", "started": time.time()})
    torch.manual_seed(config["seed"])
    base_config = BaseConfig(device="cuda", dtype="bfloat16")
    model = DecisionModel(
        BaseModel.load(base_config, local_files_only=True),
        DecisionConfig(candidate_chunk_size=config["candidate_chunk_size"]),
    )
    model.configure_training("lora")
    model.eval()
    model.save(reports / "initial-model")
    print("Evaluating initial model on 64 held-out validation examples", flush=True)
    baseline = evaluate(
        model, prepared, reports / "initial-validation", split="validation"
    )
    write_json(
        reports / "status.json",
        {"stage": "training", "started": time.time(), "baseline": baseline["overall"]},
    )
    result = train(
        prepared,
        output,
        TrainConfig(
            epochs=config["epochs"],
            effective_batch_size=config["effective_batch_size"],
            checkpoint_steps=config["checkpoint_steps"],
            seed=config["seed"],
        ),
        base_config,
        model.settings,
        model=model,
        tracking=WandbConfig(
            mode=config["wandb_mode"], name=config["name"], group="local-learning"
        ),
    )
    # Evaluate the selected saved checkpoint, not an in-memory final state.
    del model
    torch.cuda.empty_cache()
    model = DecisionModel.load(
        output / result["checkpoint"], device="cuda", dtype="bfloat16"
    )
    write_json(
        reports / "status.json",
        {"stage": "final-validation", "checkpoint": result["checkpoint"]},
    )
    final = evaluate(
        model, prepared, reports / "trained-validation", split="validation"
    )
    keys = ("accuracy", "macro_f1", "nll", "brier", "ece")
    summary = {
        "protocol": config,
        "intents": manifest["subset_protocol"]["intents"],
        "counts": manifest["counts"],
        "checkpoint": result["checkpoint"],
        "initial": {k: baseline["overall"][k] for k in keys},
        "trained": {k: final["overall"][k] for k in keys},
        "delta": {k: final["overall"][k] - baseline["overall"][k] for k in keys},
        "notes": "Small validation learning check, not test performance or calibrated confidence; no calibration fitted.",
    }
    write_json(reports / "comparison.json", summary)
    write_json(
        reports / "status.json", {"stage": "completed", "summary": "comparison.json"}
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
