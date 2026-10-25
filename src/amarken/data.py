"""Pinned dataset loaders producing a shared typed-decision schema."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from math import isclose, isfinite
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

from amarken.constants import (
    CLINC,
    CLINC_QUESTION,
    COMMERCIAL_USE,
    DATA_CACHE,
    DATA_SCHEMA_VERSION,
    DATA_SPLITS,
    DATASET_DIRECTORIES,
    DATASET_LICENSES,
    DATASET_REVISIONS,
    DATASETS,
    DECISION_KINDS,
    HUB_CACHE,
    TASKSOURCE,
)


@dataclass(frozen=True)
class DataConfig:
    dataset_id: str = CLINC
    seed: int = 42
    max_train_examples: int = 100_000
    max_per_source: int = 2_000
    shuffle_buffer: int = 10_000
    calibration_fraction: float = 0.5
    commercial_only: bool = True

    def __post_init__(self):
        if self.dataset_id not in DATASETS:
            raise ValueError(f"Supported datasets: {DATASETS}")
        if min(self.max_train_examples, self.max_per_source, self.shuffle_buffer) < 1:
            raise ValueError("Training limits and shuffle_buffer must be positive")
        if not 0 < self.calibration_fraction < 1:
            raise ValueError("calibration_fraction must be between zero and one")

    @property
    def revision(self) -> str:
        return DATASET_REVISIONS[self.dataset_id]


def validate_decision(row: dict) -> None:
    """Fail on malformed supervision rather than silently repairing labels."""
    kind, options, target = row["kind"], row["options"], row["target"]
    if (
        kind not in DECISION_KINDS
        or not isinstance(row.get("state"), str)
        or not isinstance(row.get("question"), str)
        or not row["state"].strip()
        or not row["question"].strip()
    ):
        raise ValueError(f"Invalid decision: {row['id']}")
    if (
        not isinstance(target, (list, tuple))
        or not target
        or any(
            not isinstance(p, (float, int)) or not isfinite(p) or not 0 <= p <= 1
            for p in target
        )
    ):
        raise ValueError(f"Invalid target probabilities: {row['id']}")
    if not isinstance(options, (list, tuple)) or any(
        not isinstance(option, str) or not option.strip() for option in options
    ):
        raise ValueError("Options must be nonempty string labels")
    if kind == "noul":
        if options or len(target) != 1:
            raise ValueError("noul requires no options and one p_true target")
    elif (
        len(options) < 2
        or len(options) != len(set(options))
        or any(not isinstance(option, str) or not option.strip() for option in options)
        or len(target) != len(options)
        or not isclose(sum(target), 1, abs_tol=1e-5)
    ):
        raise ValueError(f"Invalid options or distribution: {row['id']}")


def load_data(
    config: DataConfig | None = None, split: str = "train", *, curate: bool = True
) -> Iterator[dict]:
    """Stream rows; training caps apply only to Tasksource, never evaluation.

    Validation and calibration are disjoint, stable partitions of the upstream
    validation split. Tasksource group IDs keep related variants together.
    """
    config = config or DataConfig()
    if split not in DATA_SPLITS:
        raise ValueError(f"Supported splits: {DATA_SPLITS}")
    upstream_split = "validation" if split == "calibration" else split
    # Materialize pinned parquet files once, avoiding remote streaming range reads.
    directory = DATASET_DIRECTORIES[config.dataset_id]
    snapshot = snapshot_download(
        config.dataset_id,
        repo_type="dataset",
        revision=config.revision,
        cache_dir=HUB_CACHE,
        allow_patterns=f"{directory}/*.parquet",
    )
    files = sorted(
        str(path)
        for path in (Path(snapshot) / directory).glob(f"{upstream_split}-*.parquet")
    )
    if not files:
        raise ValueError(f"No parquet files for {upstream_split}")
    data = load_dataset(
        "parquet", data_files=files, split="train", streaming=True, cache_dir=DATA_CACHE
    )
    labels = data.features["intent"].names if config.dataset_id == CLINC else None
    # Add stable IDs before shuffle for CLINC, whose source has no row IDs.
    rows = enumerate(data)
    if curate and config.dataset_id == TASKSOURCE and split == "train":
        rows = enumerate(
            data.shuffle(seed=config.seed, buffer_size=config.shuffle_buffer)
        )
    counts = Counter()
    kind_counts = Counter()
    emitted = 0
    # Equal upper bounds, not a guarantee of equal availability after filtering.
    kind_limit = (config.max_train_examples + len(DECISION_KINDS) - 1) // len(
        DECISION_KINDS
    )
    for index, raw in rows:
        if labels is not None:
            identity = f"{CLINC}:{upstream_split}:{index}"
            group = sha256(raw["text"].strip().lower().encode()).hexdigest()
            row = {
                "id": identity,
                "group_id": group,
                "state": raw["text"],
                "question": CLINC_QUESTION,
                "kind": "choice",
                "options": list(labels),
                "target": [float(i == raw["intent"]) for i in range(len(labels))],
                "source": CLINC,
                "license": DATASET_LICENSES[CLINC],
                "license_use": COMMERCIAL_USE,
            }
        else:
            if config.commercial_only and raw["license_use"] != COMMERCIAL_USE:
                continue
            row = {
                key: raw[key]
                for key in (
                    "id",
                    "group_id",
                    "state",
                    "question",
                    "kind",
                    "options",
                    "target",
                    "source",
                    "license",
                    "license_use",
                )
            }
            for key in ("source_index", "question_id", "variant", "example_id"):
                if key in raw:
                    row[key] = raw[key]
            if not row["group_id"]:
                raise ValueError(f"Missing group_id: {row['id']}")
        if upstream_split == "validation":
            digest = sha256(f"{config.seed}:{row['group_id']}".encode()).digest()
            calibration = (
                int.from_bytes(digest[:8], "big") / 2**64 < config.calibration_fraction
            )
            if calibration != (split == "calibration"):
                continue
        if (
            curate
            and config.dataset_id == TASKSOURCE
            and split == "train"
            and (
                counts[row["source"]] >= config.max_per_source
                or kind_counts[row["kind"]] >= kind_limit
            )
        ):
            continue
        validate_decision(row)
        row.update(
            schema_version=DATA_SCHEMA_VERSION,
            split=split,
            dataset_id=config.dataset_id,
            revision=config.revision,
        )
        yield row
        counts[row["source"]] += 1
        kind_counts[row["kind"]] += 1
        emitted += 1
        if (
            curate
            and config.dataset_id == TASKSOURCE
            and split == "train"
            and emitted >= config.max_train_examples
        ):
            break
