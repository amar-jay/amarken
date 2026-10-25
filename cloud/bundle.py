"""Build an allowlisted source archive; never include local data or credentials."""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED = (
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "README.md",
    "AGENTS.md",
    "LICENSE",
    "cloud/setup.sh",
    "cloud/run.py",
    "cloud/bundle.py",
    "cloud/clinc.json",
    "cloud/README.md",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/cloud/amarken-source.tar.gz")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    paths = [ROOT / name for name in FIXED] + sorted(
        (ROOT / "src" / "amarken").glob("*.py")
    )
    contents = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular source file: {path}")
        contents[str(path.relative_to(ROOT))] = path.read_bytes()
    manifest = {
        name: hashlib.sha256(value).hexdigest() for name, value in contents.items()
    }
    contents["bundle-manifest.json"] = json.dumps(manifest, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as archive,
    ):
        for name, value in sorted(contents.items()):
            info = tarfile.TarInfo(name)
            info.size = len(value)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(value))
    print(
        json.dumps(
            {
                "archive": str(output),
                "files": len(contents),
                "bytes": output.stat().st_size,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
