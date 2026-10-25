"""Transactional S3-compatible checkpoint publication and verified restoration."""

import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from amarken.constants import BACKUP_FORMAT, BACKUP_SELECTORS
from amarken.storage import fingerprint, read_json, write_json

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BackupConfig:
    uri: str | None = None
    endpoint_url: str | None = None

    def __post_init__(self):
        if self.uri:
            uri = urlsplit(self.uri)
            if (
                uri.scheme != "s3"
                or not uri.netloc
                or uri.username
                or uri.password
                or uri.query
                or uri.fragment
                or ".." in PurePosixPath(uri.path).parts
            ):
                raise ValueError(
                    "Use s3://bucket/unique-run-prefix without credentials or query parameters"
                )
        if self.endpoint_url:
            endpoint = urlsplit(self.endpoint_url)
            if (
                endpoint.scheme not in ("https", "http")
                or not endpoint.netloc
                or endpoint.username
                or endpoint.password
                or endpoint.query
                or endpoint.fragment
            ):
                raise ValueError(
                    "Endpoint URL must not contain credentials or query parameters"
                )


def pack(files, output):
    """Deterministic archive containing only regular, explicitly selected files."""
    with (
        Path(output).open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as archive,
    ):
        for name, path in sorted(files.items()):
            path = Path(path)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Not a regular backup file: {path}")
            info = tarfile.TarInfo(name)
            info.size = path.stat().st_size
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            with path.open("rb") as stream:
                archive.addfile(info, stream)


def extract(archive, destination):
    with tarfile.open(archive) as stream:
        for entry in stream.getmembers():
            path = PurePosixPath(entry.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (entry.isfile() or entry.isdir())
            ):
                raise ValueError("Unsafe path or link in checkpoint archive")
        stream.extractall(destination, filter="data")


class CheckpointBackup:
    def __init__(self, config=None, *, client=None):
        self.settings = config or BackupConfig()
        self.client = client
        self.context = None
        self._uploaded = {}
        if self.settings.uri:
            uri = urlsplit(self.settings.uri)
            self.bucket, self.prefix = uri.netloc, uri.path.strip("/")
            if client is None:
                import boto3
                from botocore.config import Config

                self.client = boto3.client(
                    "s3",
                    endpoint_url=self.settings.endpoint_url,
                    config=Config(
                        retries={"mode": "standard", "total_max_attempts": 5},
                        connect_timeout=10,
                        read_timeout=60,
                        s3={"addressing_style": "path"},
                    ),
                )

    def key(self, suffix):
        return f"{self.prefix}/{suffix}" if self.prefix else suffix

    def put_json(self, key, value):
        body = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=body, ContentType="application/json"
        )
        return hashlib.sha256(body).hexdigest()

    def get_json(self, key, digest=None):
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        if digest and hashlib.sha256(body).hexdigest() != digest:
            raise ValueError("Remote snapshot manifest checksum mismatch")
        return json.loads(body)

    def upload(self, archive, category, label):
        digest = fingerprint(archive)
        key = self.key(f"{category}/{label}-{digest}.tar.gz")
        if key in self._uploaded:
            return self._uploaded[key]
        self.client.upload_file(
            str(archive), self.bucket, key, ExtraArgs={"Metadata": {"sha256": digest}}
        )
        reference = {
            "key": key,
            "sha256": digest,
            "bytes": Path(archive).stat().st_size,
        }
        self._uploaded[key] = reference
        return reference

    def prepare_context(self, output, prepared):
        if not self.settings.uri or self.context is not None:
            return
        output, prepared = Path(output), Path(prepared)
        context_dir = output / "backup"
        context_dir.mkdir(exist_ok=True)
        files = {
            f"prepared/{name}": prepared / name
            for name in (
                "manifest.json",
                *[
                    f"{split}.jsonl"
                    for split in read_json(prepared / "manifest.json")["files"]
                ],
            )
        }
        if (prepared / "exclusions.jsonl").exists():
            files["prepared/exclusions.jsonl"] = prepared / "exclusions.jsonl"
        for name in ("run.json", "data-manifest.json", "wandb-run.json"):
            if (output / name).exists():
                files[f"run/{name}"] = output / name
        source = list((ROOT / "src" / "amarken").glob("*.py"))
        source += [
            ROOT / name
            for name in (
                "pyproject.toml",
                "uv.lock",
                ".python-version",
                "README.md",
                "LICENSE",
                "AGENTS.md",
                "cloud/run.py",
                "cloud/setup.sh",
                "cloud/bundle.py",
                "cloud/clinc.json",
            )
            if (ROOT / name).is_file()
        ]
        files.update({f"source/{path.relative_to(ROOT)}": path for path in source})
        archive = context_dir / "context.tar.gz"
        pack(files, archive)
        self.context = self.upload(archive, "contexts", "recovery")
        write_json(context_dir / "context.json", self.context)
        archive.unlink()

    def checkpoint_ref(self, directory, context):
        directory = Path(directory)
        # A directory is eligible only after every file in its completion manifest verifies.
        for name, checksum in read_json(directory / "checksums.json").items():
            if fingerprint(directory / name) != checksum:
                raise ValueError(f"Incomplete checkpoint: {directory / name}")
        with tempfile.TemporaryDirectory(
            prefix="backup-", dir=directory.parent
        ) as temporary:
            archive = Path(temporary) / "checkpoint.tar.gz"
            pack(
                {
                    str(p.relative_to(directory)): p
                    for p in directory.rglob("*")
                    if p.is_file()
                },
                archive,
            )
            reference = self.upload(archive, "checkpoints", directory.name)
        return {"name": directory.name, "archive": reference, "context": context}

    def checkpoint(self, output, directory, prepared, *, step):
        if not self.settings.uri:
            return
        output = Path(output)
        state_path = output / "backup" / "status.json"
        try:
            self.prepare_context(output, prepared)
            current = self.checkpoint_ref(directory, self.context)
            best = None
            if (output / "best.json").exists():
                pointer = read_json(output / "best.json")
                best = (
                    current
                    if pointer["checkpoint"] == current["name"]
                    else self.checkpoint_ref(
                        output / pointer["checkpoint"], self.context
                    )
                )
                best = {**best, "validation_loss": pointer["validation_loss"]}
            snapshot = {
                "format": BACKUP_FORMAT,
                "step": step,
                "latest": current,
                "best": best,
            }
            identity = hashlib.sha256(
                json.dumps(snapshot, sort_keys=True).encode()
            ).hexdigest()
            key = self.key(f"snapshots/{step:07d}-{identity}.json")
            digest = self.put_json(key, snapshot)
            published = {"snapshot": key, "sha256": digest}
            # Publish only complete bundles. Interrupted multipart uploads cannot
            # replace the last valid pointer. A prefix must have exactly one writer.
            self.put_json(self.key("latest.json"), published)
            if best:
                self.put_json(self.key("best.json"), published)
            write_json(
                state_path,
                {
                    "status": "uploaded",
                    "step": step,
                    "uri": self.settings.uri,
                    "snapshot": key,
                },
            )
        except Exception as error:
            state_path.parent.mkdir(exist_ok=True)
            write_json(
                state_path, {"status": "failed", "step": step, "uri": self.settings.uri}
            )
            # Retry through botocore, then fail closed: never claim a safe backup
            # and keep training unprotected after all configured attempts fail.
            raise RuntimeError(
                "Checkpoint backup failed; local checkpoint remains intact. Check bucket access/network and resume from latest.json."
            ) from error

    def download(self, reference, destination):
        self.client.download_file(self.bucket, reference["key"], str(destination))
        if fingerprint(destination) != reference["sha256"]:
            raise ValueError("Downloaded archive checksum mismatch")

    def restore(self, output, prepared_output, *, which="latest"):
        if not self.settings.uri or which not in BACKUP_SELECTORS:
            raise ValueError("Provide a backup URI and choose latest or best")
        output, prepared_output = (
            Path(output).resolve(),
            Path(prepared_output).resolve(),
        )
        if (
            output == prepared_output
            or output in prepared_output.parents
            or prepared_output in output.parents
        ):
            raise ValueError("Run and prepared outputs must be separate directories")
        if output.exists() or prepared_output.exists():
            raise FileExistsError("Restore requires fresh output directories")
        pointer = self.get_json(self.key(f"{which}.json"))
        snapshot = self.get_json(pointer["snapshot"], pointer["sha256"])
        if snapshot["format"] != BACKUP_FORMAT or not snapshot.get(which):
            raise ValueError("Unsupported or missing backup")
        selected = snapshot[which]
        output.parent.mkdir(parents=True, exist_ok=True)
        prepared_output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="amarken-restore-", dir=output.parent
        ) as temporary:
            temporary = Path(temporary)
            self.download(selected["context"], temporary / "context.tar.gz")
            extract(temporary / "context.tar.gz", temporary / "context")
            checkpoint_dir = temporary / "run" / selected["name"]
            checkpoint_dir.mkdir(parents=True)
            self.download(selected["archive"], temporary / "checkpoint.tar.gz")
            extract(temporary / "checkpoint.tar.gz", checkpoint_dir)
            for name, checksum in read_json(checkpoint_dir / "checksums.json").items():
                if fingerprint(checkpoint_dir / name) != checksum:
                    raise ValueError("Checkpoint file checksum mismatch")
            from amarken.storage import verify_prepared

            context = temporary / "context"
            verify_prepared(context / "prepared")
            if read_json(checkpoint_dir / "run.json")["prepared_id"] != fingerprint(
                context / "prepared" / "manifest.json"
            ):
                raise ValueError("Recovery dataset differs from checkpoint")
            if (context / "run").exists():
                for path in (context / "run").iterdir():
                    shutil.copy2(path, temporary / "run" / path.name)
            # Prefer the checkpoint's exact metadata to older context copies.
            for name in ("run.json", "data-manifest.json"):
                shutil.copy2(checkpoint_dir / name, temporary / "run" / name)
            write_json(
                temporary / "run" / "latest.json", {"checkpoint": selected["name"]}
            )
            best = snapshot["best"]
            if best:
                if best["name"] != selected["name"]:
                    self.download(best["archive"], temporary / "best.tar.gz")
                    extract(temporary / "best.tar.gz", temporary / "run" / best["name"])
                write_json(
                    temporary / "run" / "best.json",
                    {
                        "checkpoint": best["name"],
                        "validation_loss": best["validation_loss"],
                    },
                )
            if (context / "source").exists():
                shutil.move(
                    str(context / "source"), temporary / "run" / "source-package"
                )
            write_json(
                temporary / "run" / "restore.json",
                {
                    "uri": self.settings.uri,
                    "selector": which,
                    "checkpoint": selected["name"],
                    "prepared_output": str(prepared_output),
                    "snapshot": pointer["snapshot"],
                },
            )
            shutil.move(str(context / "prepared"), prepared_output)
            shutil.move(str(temporary / "run"), output)
        return {
            "run": str(output),
            "prepared": str(prepared_output),
            "checkpoint": selected["name"],
        }
