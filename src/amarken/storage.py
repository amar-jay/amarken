"""Local artifacts and reproducibility metadata."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    def git(*args):
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "datasets")
        },
        "commit": git("rev-parse", "HEAD"),
        "dirty_diff": git("diff", "HEAD"),
        "untracked": git("ls-files", "--others", "--exclude-standard"),
        "source_snapshot": {
            str(path): path.read_text() for path in Path("src/amarken").glob("*.py")
        },
        "dependency_lock_sha256": fingerprint("uv.lock")
        if Path("uv.lock").exists()
        else None,
    }


def verify_prepared(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    for split, expected in manifest["files"].items():
        if fingerprint(directory / f"{split}.jsonl") != expected:
            raise ValueError(f"Prepared data changed: {split}")
    return manifest


def refresh_checksums(directory):
    directory = Path(directory)
    write_json(
        directory / "checksums.json",
        {
            str(p.relative_to(directory)): fingerprint(p)
            for p in directory.rglob("*")
            if p.is_file() and p.name != "checksums.json"
        },
    )
