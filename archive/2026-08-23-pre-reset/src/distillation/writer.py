import json
from pathlib import Path
from typing import Any
from collections import Counter
import time

class ShardWriter:
    def __init__(self, output_dir: Path, shard_size: int):
        self.output_dir = output_dir
        self.shards = output_dir / "shards"
        self.shards.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.accepted = 0
        self.hashes: set[str] = set()
        self.rows: list[dict[str, Any]] = []
        self.max_index = -1
        for path in sorted(self.shards.glob("shard-*.jsonl")):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self.accepted += 1
                self.hashes.add(row["content_sha256"])
                self.max_index = max(self.max_index, int(row["id"].split("-")[-1]))
        partial = self.shards / "current.partial.jsonl"
        if partial.exists():
            for line in partial.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows.append(row)
                    self.accepted += 1
                    self.hashes.add(row["content_sha256"])
                    self.max_index = max(self.max_index, int(row["id"].split("-")[-1]))

    def add(self, row: dict[str, Any]) -> bool:
        if row["content_sha256"] in self.hashes:
            return False
        self.hashes.add(row["content_sha256"])
        self.rows.append(row)
        self.accepted += 1
        if len(self.rows) >= self.shard_size:
            self.commit()
        elif len(self.rows) % 25 == 0:
            self.sync_partial()
        return True

    def sync_partial(self):
        path = self.shards / "current.partial.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in self.rows
            )
        )

    def commit(self):
        if not self.rows:
            return
        index = len(list(self.shards.glob("shard-*.jsonl")))
        temp = self.shards / "current.partial.jsonl"
        self.sync_partial()
        temp.replace(self.shards / f"shard-{index:06d}.jsonl")
        self.rows = []


def write_progress(
    output_dir: Path,
    config: dict[str, Any],
    writer: ShardWriter,
    attempted: int,
    rejections: Counter,
    start: float,
    identity: dict[str, Any],
):
    progress = {
        "schema_version": 1,
        "target_accepted": config["target_accepted"],
        "accepted": writer.accepted,
        "attempted_specs": attempted,
        "completion": writer.accepted / config["target_accepted"],
        "elapsed_seconds": round(time.monotonic() - start, 1),
        "accepted_per_second": round(
            writer.accepted / max(time.monotonic() - start, 0.001), 3
        ),
        "rejections": dict(rejections),
        "model": identity,
        "updated_at_unix": time.time(),
    }
    temp = output_dir / "progress.tmp"
    temp.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
    temp.replace(output_dir / "progress.json")