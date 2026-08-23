"""Build a deterministic, provenance-preserving EN/TR/code proxy corpus.

The builder intentionally uses only the Python standard library: preprocessing
must remain auditable and runnable before the eventual training stack is chosen.
Every emitted document retains its source ID, source-relative locator, language,
domain, split group and content hash. Distillation outputs are not accepted by
the source schema, keeping this first proxy corpus purely source text and code.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
from pathlib import Path
import re
import sysconfig
import time
import unicodedata
from typing import Iterable, Iterator

from ftfy import fix_encoding
from ftfy.badness import badness

TOKEN = re.compile(r"[\w]+|[^\w\s]", re.UNICODE)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE = re.compile(r"[ \t\r\n]+")
# These characters are normal Unicode in some languages, but not in this
# EN/TR corpus. After ftfy has attempted a conservative encoding-only repair,
# their survival in prose is strong evidence of an irrecoverably damaged row.
RESIDUAL_MOJIBAKE = re.compile(r"[\u0080-\u009fÃÄÅ]")


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    language: str
    domain: str
    source_id: str
    locator: str
    group_id: str
    content_sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(text: str, *, code: bool) -> str:
    # NFKC removes compatibility-only spelling differences that otherwise evade
    # deduplication; Turkish letters and ordinary source-code syntax are stable.
    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = CONTROL.sub("", text)
    if code:
        # Indentation/newlines are semantic in Python. Trailing horizontal space
        # is not, so remove it while preserving blank lines and leading columns.
        return "\n".join(line.rstrip() for line in text.split("\n")).strip()
    # OPUS records are sentence-like documents. Canonicalizing internal spacing
    # makes exact duplicate detection insensitive to transport whitespace.
    return SPACE.sub(" ", text).strip()


def repair_text_encoding(text: str) -> tuple[str | None, str]:
    """Repair recoverable mojibake; reject suspicious residual prose.

    ftfy's encoding-only path preserves typography and HTML literally, unlike
    its broader ``fix_text`` operation. Repair must precede NFKC because NFKC
    decomposes mojibake such as ``Ã¼`` into ``Ã1⁄4``, destroying the byte pattern
    needed to recover ``ü``. A row is accepted as repaired only when ftfy's
    badness score strictly improves. Residual C1 controls or UTF-8-as-Latin-1
    lead characters are quarantined rather than guessed.
    """
    original_badness = badness(text)
    repaired = fix_encoding(text)
    if repaired != text and badness(repaired) < original_badness:
        text = repaired
        status = "repaired"
    else:
        status = "unchanged"
    if RESIDUAL_MOJIBAKE.search(text):
        return None, "rejected_residual_mojibake"
    return text, status


def _canonical(text: str) -> str:
    # Casefolding is used only for duplicate/contamination fingerprints; emitted
    # text retains original case, including the distinct Turkish dotted letters.
    return SPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _stable_hash(seed: str, value: str) -> str:
    return _sha256_bytes(f"{seed}\0{value}".encode("utf-8"))


def _source_paths(root: Path, source: dict) -> tuple[list[Path], Path]:
    """Resolve input paths and the base used for stable source locators."""
    kind, pattern = source["kind"], source["path"]
    if kind == "lines":
        return [root / pattern], root
    if kind == "glob":
        return sorted(root.glob(pattern)), root
    if kind == "python_stdlib":
        # Resolve the active interpreter's stdlib rather than site-packages. The
        # resolved files and hashes are recorded, so environment drift is visible.
        base = Path(sysconfig.get_path("stdlib"))
        excluded = set(source.get("exclude_directories", ()))
        paths = [
            path
            for path in base.rglob("*.py")
            if not excluded.intersection(path.relative_to(base).parts)
        ]
        return sorted(paths), base
    raise ValueError(f"unsupported source kind: {kind}")


def _iter_source(
    root: Path,
    source: dict,
    seed: str,
    encoding_quality: Counter,
    encoding_rejections: list[dict],
) -> Iterator[Document]:
    kind, _ = source["kind"], source["path"]
    paths, source_base = _source_paths(root, source)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if kind == "lines":
            stream: Iterable[tuple[int, str]]
            with path.open("r", encoding="utf-8", errors="strict") as handle:
                stream = enumerate(handle, 1)
                for line_number, raw in stream:
                    raw_text = raw.rstrip("\n")
                    repaired, repair_status = repair_text_encoding(raw_text)
                    encoding_quality[(source["id"], repair_status)] += 1
                    if repaired is None:
                        encoding_rejections.append(
                            {
                                "id": _stable_hash(
                                    seed,
                                    f"encoding-rejection:{source['id']}:{line_number}",
                                ),
                                "source_id": source["id"],
                                "locator": f"{path.relative_to(root)}:{line_number}",
                                "language": source["language"],
                                "domain": source["domain"],
                                "reason": repair_status,
                                "text": raw_text,
                            }
                        )
                        continue
                    text = _normalize(repaired, code=False)
                    if not text:
                        continue
                    locator = f"{path.relative_to(root)}:{line_number}"
                    group = f"{source['group_namespace']}:{line_number}"
                    content_hash = _sha256_bytes(_canonical(text).encode("utf-8"))
                    yield Document(
                        id=_stable_hash(
                            seed, f"{source['id']}:{line_number}:{content_hash}"
                        ),
                        text=text,
                        language=source["language"],
                        domain=source["domain"],
                        source_id=source["id"],
                        locator=locator,
                        group_id=group,
                        content_sha256=content_hash,
                    )
        else:
            text = _normalize(
                path.read_text(encoding="utf-8", errors="strict"), code=True
            )
            if not text:
                continue
            relative = str(path.relative_to(source_base))
            # External stdlib locators name the upstream tree rather than leaking
            # a machine-specific pyenv installation path into document IDs.
            locator = f"cpython/Lib/{relative}" if kind == "python_stdlib" else relative
            content_hash = _sha256_bytes(_canonical(text).encode("utf-8"))
            yield Document(
                id=_stable_hash(seed, f"{source['id']}:{locator}:{content_hash}"),
                text=text,
                language=source["language"],
                domain=source["domain"],
                source_id=source["id"],
                locator=locator,
                group_id=f"{source['group_namespace']}:{locator}",
                content_sha256=content_hash,
            )


def _sample(
    source_docs: Iterable[Document], limit: int | None, seed: str
) -> list[Document]:
    if limit is None:
        return list(source_docs)
    # Retaining the lowest seeded hashes is invariant to input order and avoids
    # the first-N/domain-order bias of large web or translation corpora.
    heap: list[tuple[int, str, Document]] = []
    for document in source_docs:
        rank = int(_stable_hash(seed, document.id), 16)
        item = (-rank, document.id, document)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]


def _token_strings(text: str) -> list[str]:
    return TOKEN.findall(_canonical(text))


def _simhash(text: str) -> int:
    tokens = _token_strings(text)
    # Token trigrams retain local word order; short records fall back to their
    # complete token sequence so they still receive a meaningful fingerprint.
    features = (
        "\x1f".join(tokens[index : index + 3])
        for index in range(max(1, len(tokens) - 2))
    )
    weights = [0] * 64
    for feature in features:
        value = int.from_bytes(
            hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, weight in enumerate(weights) if weight >= 0)


def _deduplicate(
    documents: Iterable[Document], max_hamming: int
) -> tuple[list[Document], Counter]:
    exact: set[tuple[str, str]] = set()
    # Four 16-bit bands make Hamming-near candidate lookup subquadratic. Final
    # distance verification prevents a shared band alone from deleting a record.
    bands: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    fingerprints: list[int] = []
    kept: list[Document] = []
    removed: Counter = Counter()
    for document in sorted(documents, key=lambda value: value.id):
        namespace = document.language  # Never collapse translations across languages.
        exact_key = (namespace, document.content_sha256)
        if exact_key in exact:
            removed["exact_duplicate"] += 1
            continue
        fingerprint = _simhash(document.text)
        candidate_indexes: set[int] = set()
        for band in range(4):
            candidate_indexes.update(
                bands[(namespace, band, (fingerprint >> (band * 16)) & 0xFFFF)]
            )
        if any(
            (fingerprint ^ fingerprints[index]).bit_count() <= max_hamming
            for index in candidate_indexes
        ):
            removed["near_duplicate"] += 1
            continue
        index = len(fingerprints)
        exact.add(exact_key)
        fingerprints.append(fingerprint)
        kept.append(document)
        for band in range(4):
            bands[(namespace, band, (fingerprint >> (band * 16)) & 0xFFFF)].append(
                index
            )
    return kept, removed


def _reference_fingerprints(
    root: Path, patterns: list[str], ngram_size: int
) -> tuple[set[str], set[int], list[dict]]:
    exact, ngrams, manifest = set(), set(), []
    paths = sorted(
        {path for pattern in patterns for path in root.glob(pattern) if path.is_file()}
    )
    for path in paths:
        text = _normalize(
            path.read_text(encoding="utf-8", errors="strict"), code=path.suffix == ".py"
        )
        canonical = _canonical(text)
        exact.add(_sha256_bytes(canonical.encode("utf-8")))
        tokens = _token_strings(canonical)
        for index in range(max(0, len(tokens) - ngram_size + 1)):
            ngram = "\x1f".join(tokens[index : index + ngram_size]).encode("utf-8")
            ngrams.add(
                int.from_bytes(hashlib.blake2b(ngram, digest_size=8).digest(), "big")
            )
        manifest.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return exact, ngrams, manifest


def _is_contaminated(
    document: Document, exact: set[str], ngrams: set[int], size: int
) -> bool:
    if document.content_sha256 in exact:
        return True
    tokens = _token_strings(document.text)
    # A 13-token exact window is long enough to avoid ordinary phrase matches,
    # yet catches benchmark questions embedded inside a larger training record.
    for index in range(max(0, len(tokens) - size + 1)):
        raw = "\x1f".join(tokens[index : index + size]).encode("utf-8")
        if (
            int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")
            in ngrams
        ):
            return True
    return False


def _split(seed: str, group_id: str, validation_basis_points: int) -> str:
    bucket = int(_stable_hash(seed, group_id)[:8], 16) % 10_000
    return "validation" if bucket < validation_basis_points else "train"


def build(config_path: Path, output_dir: Path, root: Path = Path(".")) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not 0 < config["validation_basis_points"] < 10_000:
        raise ValueError("validation_basis_points must be between 1 and 9999")
    started = time.perf_counter()
    source_manifest, documents = [], []
    encoding_quality: Counter = Counter()
    encoding_rejections: list[dict] = []
    for source in config["sources"]:
        path_matches, source_base = _source_paths(root, source)
        path_matches = [path for path in path_matches if path.is_file()]
        if not path_matches:
            raise FileNotFoundError(source["path"])
        sampled = _sample(
            _iter_source(
                root, source, config["seed"], encoding_quality, encoding_rejections
            ),
            source.get("max_documents"),
            config["seed"],
        )
        documents.extend(sampled)
        license_paths = (
            [source_base / "LICENSE.txt"] if source["kind"] == "python_stdlib" else []
        )
        source_manifest.append(
            {
                **source,
                "input_files": [
                    {
                        "path": (
                            f"cpython/Lib/{path.relative_to(source_base)}"
                            if source["kind"] == "python_stdlib"
                            else str(path.relative_to(root))
                        ),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                    for path in path_matches
                ],
                "license_files": [
                    {
                        "path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                    for path in license_paths
                    if path.is_file()
                ],
                "sampled_documents": len(sampled),
            }
        )
    deduplicated, removed = _deduplicate(
        documents, config["near_duplicate_hamming_distance"]
    )
    exact_refs, ngram_refs, reference_manifest = _reference_fingerprints(
        root, config["contamination_references"], config["contamination_ngram_tokens"]
    )
    clean, contaminated = [], []
    for document in deduplicated:
        (
            contaminated
            if _is_contaminated(
                document, exact_refs, ngram_refs, config["contamination_ngram_tokens"]
            )
            else clean
        ).append(document)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    handles = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "validation")
    }
    try:
        for document in sorted(clean, key=lambda value: value.id):
            split = _split(
                config["seed"], document.group_id, config["validation_basis_points"]
            )
            handles[split].write(
                json.dumps(asdict(document), ensure_ascii=False, sort_keys=True) + "\n"
            )
            counts[(split, document.language)] += 1
    finally:
        for handle in handles.values():
            handle.close()
    contamination_path = output_dir / "contamination.jsonl"
    with contamination_path.open("w", encoding="utf-8") as handle:
        for document in sorted(contaminated, key=lambda value: value.id):
            handle.write(
                json.dumps(
                    {**asdict(document), "reason": "reference_ngram_or_exact_match"},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    encoding_rejections_path = output_dir / "encoding_rejections.jsonl"
    with encoding_rejections_path.open("w", encoding="utf-8") as handle:
        for rejection in sorted(encoding_rejections, key=lambda value: value["id"]):
            handle.write(
                json.dumps(rejection, ensure_ascii=False, sort_keys=True) + "\n"
            )
    manifest = {
        "schema_version": config["schema_version"],
        "seed": config["seed"],
        "config_sha256": _sha256_file(config_path),
        "distillation": False,
        "policies": {
            "encoding_quality": "ftfy encoding-only repair before normalization; residual prose mojibake rejected",
            "normalization": "NFKC after encoding repair; prose whitespace collapsed; code indentation/newlines retained",
            "sampling": "lowest seeded document hashes per source",
            "deduplication": f"language-local canonical exact hash plus 64-bit trigram SimHash <= {config['near_duplicate_hamming_distance']}",
            "split": f"group-hash; {config['validation_basis_points']} validation basis points",
            "contamination": f"canonical exact document or matching {config['contamination_ngram_tokens']}-token reference window",
        },
        "sources": source_manifest,
        "contamination_references": reference_manifest,
        "input_sampled_documents": len(documents),
        "deduplicated_documents": len(deduplicated),
        "clean_documents": len(clean),
        "contaminated_documents": len(contaminated),
        "removed": dict(removed),
        "raw_line_encoding_quality": {
            source_id: {
                status: encoding_quality[(source_id, status)]
                for status in ("unchanged", "repaired", "rejected_residual_mojibake")
                if encoding_quality[(source_id, status)]
            }
            for source_id in sorted(
                {source_id for source_id, _status in encoding_quality}
            )
        },
        "split_language_counts": {
            f"{split}:{language}": count
            for (split, language), count in sorted(counts.items())
        },
        "outputs": {},
        "elapsed_seconds": time.perf_counter() - started,
    }
    for name in (
        "train.jsonl",
        "validation.jsonl",
        "contamination.jsonl",
        "encoding_rejections.jsonl",
    ):
        path = output_dir / name
        manifest["outputs"][name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "lines": sum(1 for _ in path.open("rb")),
        }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/proxy_dataset.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/proxy-v1")
    )
    args = parser.parse_args()
    report = build(args.config, args.output_dir)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "clean_documents",
                    "contaminated_documents",
                    "removed",
                    "split_language_counts",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
