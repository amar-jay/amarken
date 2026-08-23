"""High-throughput offline vLLM generator for a single GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from src.distillation.synthetic_pretraining import (
    ShardWriter,
)
from src.distillation.grounded_pilot import (
  ROOT,
  valid, 
  hash_fraction,
  make_spec,
  GENERATOR_SYSTEM
 )


def generation_conversation(spec: dict[str, Any]) -> list[dict[str, str]]:
    language = {
        "en": "English",
        "tr": "Turkish",
        "bilingual": "English followed by Turkish",
    }[spec["language"]]
    grounding = (
        (
            " You must preserve these already-verified values or names: "
            + ", ".join(spec["required"])
            + ". Do not recompute or alter them."
        )
        if spec["required"]
        else ""
    )
    return [
        {
            "role": "system",
            "content": GENERATOR_SYSTEM
            + f" Required output language: {language}."
            + grounding,
        },
        *spec["messages"][1:],
    ]


def sampling_kwargs(config: dict[str, Any], seed: int, retry: bool) -> dict[str, Any]:
    return {
        "temperature": 0.0 if retry else config["temperature"],
        "top_p": config["top_p"],
        "top_k": config["top_k"],
        "presence_penalty": config["presence_penalty"],
        "repetition_penalty": config["repetition_penalty"],
        "max_tokens": config["max_tokens"],
        "seed": seed,
    }


def generate_batch(
    llm: Any,
    sampling_params_type: Callable[..., Any],
    config: dict[str, Any],
    specs: list[dict[str, Any]],
    attempts: dict[str, int],
):
    conversations = [generation_conversation(spec) for spec in specs]
    params = [
        sampling_params_type(
            **sampling_kwargs(
                config,
                config["seed"] + int(spec["id"].split("-")[-1]) + attempts[spec["id"]],
                attempts[spec["id"]] > 0,
            )
        )
        for spec in specs
    ]
    outputs = llm.chat(
        conversations,
        sampling_params=params,
        use_tqdm=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    result = []
    for spec, output in zip(specs, outputs, strict=True):
        candidate = output.outputs[0]
        metrics = {
            "finish_reason": candidate.finish_reason,
            "stop_reason": candidate.stop_reason,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(candidate.token_ids),
        }
        result.append((spec, candidate.text.strip(), metrics))
    return result


def write_progress(
    output_dir: Path,
    config: dict[str, Any],
    writer: ShardWriter,
    attempted: int,
    rejected: Counter,
    started: float,
    initial: int,
):
    elapsed = max(time.monotonic() - started, 0.001)
    progress = {
        "schema_version": 1,
        "backend": "vllm-offline",
        "model": config["model"],
        "target_accepted": config["target_accepted"],
        "accepted": writer.accepted,
        "attempted_specs": attempted,
        "completion": writer.accepted / config["target_accepted"],
        "elapsed_seconds_this_run": round(elapsed, 2),
        "accepted_per_second_this_run": round((writer.accepted - initial) / elapsed, 3),
        "rejections_this_run": dict(rejected),
        "updated_at_unix": time.time(),
    }
    temporary = output_dir / "progress.tmp"
    temporary.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(output_dir / "progress.json")


def run(
    config_path: Path,
    max_new: int | None = None,
    output_override: Path | None = None,
    llm: Any | None = None,
    sampling_params_type: Callable[..., Any] | None = None,
):
    config = json.loads(config_path.read_text())
    output_dir = output_override or ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if llm is None or sampling_params_type is None:
        from vllm import LLM, SamplingParams


        config.tensor_parallel_size = getattr(config, "tensor_parallel_size", 1)
        config.trust_remote_code = getattr(config, "trust_remote_code", False)
        llm = LLM(**config)
        sampling_params_type = SamplingParams
    writer = ShardWriter(output_dir, config["shard_size"])
    initial = writer.accepted
    next_index = writer.max_index + 1
    attempted = 0
    rejected: Counter = Counter()
    started = time.monotonic()
    with (output_dir / "rejections.jsonl").open("a") as rejection_file:
        while writer.accepted < config["target_accepted"] and (
            max_new is None or writer.accepted - initial < max_new
        ):
            remaining = config["target_accepted"] - writer.accepted
            if max_new is not None:
                remaining = min(remaining, max_new - (writer.accepted - initial))
            count = min(config["request_batch_size"], remaining)
            pending = [make_spec(next_index + i, config["seed"]) for i in range(count)]
            next_index += count
            attempts = {spec["id"]: 0 for spec in pending}
            attempted += len(pending)
            for attempt in range(config["max_attempts"]):
                if not pending:
                    break
                retry = []
                for spec, text, metrics in generate_batch(
                    llm, sampling_params_type, config, pending, attempts
                ):
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
                        record = {
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
                                "backend": "vllm-offline",
                                "model": config["model"],
                                "seed": config["seed"]
                                + int(spec["id"].split("-")[-1])
                                + attempt,
                                "attempt": attempt,
                                "metrics": metrics,
                            },
                        }
                        if not writer.add(record):
                            ok, reason = False, "exact_duplicate"
                    if not ok:
                        rejected[reason] += 1
                        rejection_file.write(
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
                        attempts[spec["id"]] += 1
                        retry.append(spec)
                rejection_file.flush()
                pending = retry
            write_progress(
                output_dir, config, writer, attempted, rejected, started, initial
            )
            rate = (writer.accepted - initial) / max(time.monotonic() - started, 0.001)
            print(
                f"accepted={writer.accepted:,}/{config['target_accepted']:,} rate={rate:.2f}/s pending_after_retries={len(pending)}",
                flush=True,
            )
    writer.sync_partial()
    write_progress(output_dir, config, writer, attempted, rejected, started, initial)
    manifest = {
        "schema_version": 1,
        "backend": "vllm-offline",
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "accepted": writer.accepted,
        "complete": writer.accepted >= config["target_accepted"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/data-generation/synthetic/pretraining/a100-1m.json",
    )
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.config, args.max_new, args.output_dir)


if __name__ == "__main__":
    main()
