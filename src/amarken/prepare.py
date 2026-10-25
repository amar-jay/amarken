"""Prepare auditable local partitions; no model weights are required."""

import json
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from amarken.constants import (
    DATA_SCHEMA_VERSION,
    DATA_SPLITS,
    PROMPT_VERSION,
    TASKSOURCE,
)
from amarken.data import DataConfig, load_data, validate_decision
from amarken.decision import render_candidates
from amarken.storage import environment, fingerprint, write_json


@dataclass(frozen=True)
class PrepareConfig:
    max_length: int = 2048
    unseen_fraction: float = 0.1
    # Explicitly reviewed exact source names; family overrides join known aliases.
    reviewed_sources: tuple[str, ...] = ()
    source_families: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.max_length < 1 or not 0 < self.unseen_fraction < 1:
            raise ValueError("Invalid length or unseen fraction")


def normalized_state(state):
    return " ".join(unicodedata.normalize("NFKC", state).casefold().split())


def fraction(text):
    return int.from_bytes(sha256(text.encode()).digest()[:8], "big") / 2**64


def prepare(
    output,
    tokenizer,
    data_config=None,
    config=None,
    *,
    partitions=None,
    tokenizer_metadata=None,
):
    data_config = data_config or DataConfig()
    config = config or PrepareConfig()
    if data_config.dataset_id == TASKSOURCE and not config.reviewed_sources:
        raise ValueError("Tasksource requires a reviewed source allowlist")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    database = sqlite3.connect(output / "staging.sqlite")
    database.execute(
        "CREATE TABLE rows (id TEXT PRIMARY KEY, root TEXT, split TEXT, payload TEXT)"
    )
    parent, state_owner, roots_by_split, families = (
        {},
        {},
        {},
        dict(config.source_families),
    )
    exclusions, counts = Counter(), Counter()
    excluded = (output / "exclusions.jsonl").open("w")

    def reject(row, reason):
        exclusions[reason] += 1
        excluded.write(
            json.dumps(
                {"id": row.get("id"), "source": row.get("source"), "reason": reason}
            )
            + "\n"
        )

    def root(key):
        parent.setdefault(key, key)
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    heldout = set()
    sources = set()
    try:
        for split in DATA_SPLITS:
            rows = (
                partitions.get(split, [])
                if partitions is not None
                else load_data(data_config, split, curate=False)
            )
            for row in rows:
                try:
                    validate_decision(row)
                except (ValueError, KeyError, TypeError):
                    reject(row, "invalid_schema")
                    continue
                source = row["source"]
                family = families.get(source, source)
                if data_config.dataset_id == TASKSOURCE:
                    if "clinc" in source.casefold() or "clinc" in family.casefold():
                        reject(row, "clinc_source_excluded")
                        continue
                    if source not in config.reviewed_sources:
                        reject(row, "source_not_reviewed")
                        continue
                    if (
                        data_config.commercial_only
                        and row.get("license_use") != "commercial"
                    ):
                        reject(row, "license_filter")
                        continue
                    if (
                        fraction(f"{data_config.seed}:{family}")
                        < config.unseen_fraction
                    ):
                        heldout.add(family)
                        if split != "test":
                            reject(row, "heldout_source_non_test")
                            continue
                        split_name = "unseen_test"
                    else:
                        split_name = split
                else:
                    split_name = split
                texts = render_candidates(row["state"], row)
                lengths = [
                    len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]
                ]
                if max(lengths) > config.max_length:
                    reject(row, "oversized")
                    continue
                key = f"{source}:{row['group_id']}"
                state = sha256(normalized_state(row["state"]).encode()).hexdigest()
                root(key)
                if state in state_owner:
                    union(key, state_owner[state])
                else:
                    state_owner[state] = key
                row = {
                    **row,
                    "split": split_name,
                    "source_family": family,
                    "max_prompt_tokens": max(lengths),
                }
                try:
                    database.execute(
                        "INSERT INTO rows VALUES (?, ?, ?, ?)",
                        (row["id"], key, split_name, json.dumps(row)),
                    )
                except sqlite3.IntegrityError:
                    reject(row, "duplicate_id")
                sources.add(source)
        database.commit()
        priority = {
            "train": 0,
            "validation": 1,
            "calibration": 2,
            "test": 3,
            "unseen_test": 4,
        }
        for key, split in database.execute("SELECT root, split FROM rows"):
            canonical = root(key)
            previous = roots_by_split.get(canonical, "train")
            roots_by_split[canonical] = max((previous, split), key=priority.__getitem__)
        splits = (*DATA_SPLITS, "unseen_test")
        handles = {split: (output / f"{split}.jsonl").open("w") for split in splits}
        selected, kind_counts, source_counts, duplicate_decisions = (
            Counter(),
            Counter(),
            Counter(),
            set(),
        )
        # Stable hash ordering, independent of remote iteration/buffer order.
        database.create_function(
            "ordering",
            1,
            lambda identity: sha256(
                f"{data_config.seed}:{identity}".encode()
            ).hexdigest(),
        )
        try:
            for key, split, payload in database.execute(
                "SELECT root, split, payload FROM rows ORDER BY ordering(id)"
            ):
                row = json.loads(payload)
                if roots_by_split[root(key)] != split:
                    reject(row, "overlapping_group_or_state")
                    continue
                identity = sha256(
                    json.dumps(
                        [
                            normalized_state(row["state"]),
                            row["question"],
                            row["kind"],
                            row["options"],
                            row["target"],
                        ],
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                if identity in duplicate_decisions:
                    reject(row, "duplicate_decision")
                    continue
                duplicate_decisions.add(identity)
                if split == "train" and data_config.dataset_id == TASKSOURCE:
                    limit = (data_config.max_train_examples + 2) // 3
                    if (
                        selected[split] >= data_config.max_train_examples
                        or source_counts[row["source"]] >= data_config.max_per_source
                        or kind_counts[row["kind"]] >= limit
                    ):
                        reject(row, "training_cap")
                        continue
                handles[split].write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
                selected[split] += 1
                if split == "train":
                    source_counts[row["source"]] += 1
                    kind_counts[row["kind"]] += 1
                counts[f"{split}:{row['source']}:{row['kind']}"] += 1
                # Targets can be soft; retain mass rather than inventing hard labels.
                for option, mass in zip(row["options"] or ["p_true"], row["target"]):
                    counts[
                        f"target_mass:{split}:{row['source']}:{row['kind']}:{option}"
                    ] += mass
        finally:
            for handle in handles.values():
                handle.close()
        if (
            not selected["train"]
            or not selected["validation"]
            or not selected["calibration"]
        ):
            raise ValueError(
                "Preparation produced an empty train, validation or calibration split; inspect exclusions"
            )
        excluded.flush()
        manifest = {
            "schema_version": DATA_SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "data_config": asdict(data_config),
            "prepare_config": asdict(config),
            "revision": data_config.revision,
            "tokenizer": tokenizer_metadata,
            "counts": dict(selected),
            "source_kind_counts": dict(counts),
            "exclusions": dict(exclusions),
            "reviewed_sources": list(config.reviewed_sources),
            "observed_sources": sorted(sources),
            "heldout_families": sorted(heldout),
            "files": {
                split: fingerprint(output / f"{split}.jsonl") for split in splits
            },
            "exclusion_fingerprint": fingerprint(output / "exclusions.jsonl"),
            "environment": environment(),
            "status": "prepared; semantic near-duplicates and unknown source aliases require review",
        }
        write_json(output / "manifest.json", manifest)
        return manifest
    finally:
        excluded.close()
        database.close()
        (output / "staging.sqlite").unlink(missing_ok=True)
