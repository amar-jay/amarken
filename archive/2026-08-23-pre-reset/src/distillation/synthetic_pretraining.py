"""Resumable million-sample, no-code EN/TR conversational pretraining generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.distillation.grounded_pilot import (
    hash_fraction,
    ROOT,
    valid,
    model_identity,
    ollama_json,
    GENERATOR_SYSTEM,
    make_spec,
)

from src.distillation.writer import ShardWriter, write_progress




def infer(
    config: dict[str, Any], spec: dict[str, Any], attempt: int
) -> tuple[str, dict[str, Any]]:
    language = {
        "en": "English",
        "tr": "Turkish",
        "bilingual": "English followed by Turkish",
    }[spec["language"]]
    grounding = (
        (
            " You must preserve these already-verified values or names in the answer: "
            + ", ".join(spec["required"])
            + ". Do not recompute or alter them."
        )
        if spec["required"]
        else ""
    )
    messages = [
        {
            "role": "system",
            "content": GENERATOR_SYSTEM
            + f" Required output language: {language}."
            + grounding,
        },
        *spec["messages"][1:],
    ]
    payload = {
        "model": config["model"],
        "stream": False,
        "think": False,
        "messages": messages,
        "options": {
            "seed": config["seed"] + int(spec["id"].split("-")[-1]) + attempt,
            "temperature": config["temperature"] if attempt == 0 else 0,
            "num_ctx": config["num_ctx"],
            "num_predict": config["num_predict"],
        },
        "keep_alive": -1,
    }
    response = ollama_json(config["base_url"], "/api/chat", payload)
    metrics = {
        key: response.get(key)
        for key in (
            "created_at",
            "done_reason",
            "total_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        )
    }
    return (response.get("message", {}).get("content") or "").strip(), metrics




def run(
    config_path: Path, max_new: int | None = None, output_override: Path | None = None
):
    config = json.loads(config_path.read_text())
    output_dir = output_override or ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = model_identity(config["base_url"], config["model"])
    writer = ShardWriter(output_dir, config["shard_size"])
    initial = writer.accepted
    attempted = writer.accepted
    rejects = Counter()
    start = time.monotonic()
    reject_path = output_dir / "rejections.jsonl"
    last_reported = writer.accepted
    with reject_path.open("a") as reject_file:
        index = writer.max_index + 1
        while writer.accepted < config["target_accepted"] and (
            max_new is None or writer.accepted - initial < max_new
        ):
            spec = make_spec(index, config["seed"])
            for attempt in range(config["max_attempts"]):
                try:
                    text, metrics = infer(config, spec, attempt)
                except Exception as exc:
                    reason = f"request_error:{type(exc).__name__}"
                    rejects[reason] += 1
                    time.sleep(min(2**attempt, 8))
                    continue
                ok, reason = valid(text, spec, config)
                if ok:
                    messages = [
                        *spec["messages"],
                        {"role": "assistant", "content": text},
                    ]
                    digest = hashlib.sha256(
                        json.dumps(
                            messages, ensure_ascii=False, sort_keys=True
                        ).encode()
                    ).hexdigest()
                    row = {
                        "id": spec["id"],
                        "group": spec["group"],
                        "split": (
                            "validation"
                            if hash_fraction(spec["group"])
                            < config["validation_fraction"]
                            else "train"
                        ),
                        "category": spec["category"],
                        "language": spec["language"],
                        "messages": messages,
                        "content_sha256": digest,
                        "generation": {
                            "seed": config["seed"] + index + attempt,
                            "attempt": attempt,
                            "metrics": metrics,
                        },
                    }
                    if writer.add(row):
                        break
                    reason = "exact_duplicate"
                rejects[reason] += 1
                reject_file.write(
                    json.dumps(
                        {
                            "id": spec["id"],
                            "attempt": attempt,
                            "reason": reason,
                            "output": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                reject_file.flush()
            attempted += 1
            index += 1
            if writer.accepted - last_reported >= 100:
                last_reported = writer.accepted
                write_progress(
                    output_dir, config, writer, attempted, rejects, start, identity
                )
                print(
                    f"accepted={writer.accepted:,}/{config['target_accepted']:,} rate={(writer.accepted-initial)/max(time.monotonic()-start,.001):.2f}/s",
                    flush=True,
                )
    writer.sync_partial()
    write_progress(output_dir, config, writer, attempted, rejects, start, identity)
    manifest = {
        "schema_version": 1,
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model": identity,
        "accepted": writer.accepted,
        "shard_size": config["shard_size"],
        "complete": writer.accepted >= config["target_accepted"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/data-generation/synthetic/pretraining/local-1m.json"
    )
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.config, args.max_new, args.output_dir)


if __name__ == "__main__":
    main()
