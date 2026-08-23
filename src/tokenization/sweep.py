"""Train and gate balanced EN/TR/code tokenizer candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Iterable
from itertools import islice

from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
)

from src.data.proxy import repair_text_encoding
from src.tokenization.tokenizer import AmarkenTokenizer, SPECIAL_TOKENS
from src.tokenization.visualize import _sample_text, dataset_files

# cl100k-style splitting keeps contractions, short digit groups, punctuation,
# newlines, and horizontal whitespace in learnable regions before byte BPE.
TIKTOKEN_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|"
    r" ?[^\s\p{L}\p{N}]++[\r\n]*+|\s*[\r\n]|\s+(?!\S)|\s+"
)

# Turkish proper nouns separate productive suffixes with an apostrophe. This
# branch must precede English 'd/'t contractions or Ankara'da becomes 'd + a.
# Longest forms come first so İstanbul'dan is not prematurely split as 'da + n.
TURKISH_APOSTROPHE_SUFFIX = (
    r"(?i:[’'](?:larımızdan|lerimizden|larımız|lerimiz|"
    r"dan|den|tan|ten|dır|dir|dur|dür|tır|tir|tur|tür|"
    r"nın|nin|nun|nün|yı|yi|yu|yü|ya|ye|da|de|ta|te|"
    r"ın|in|un|ün|la|le|lı|li|lu|lü|a|e))"
)
TIKTOKEN_TURKISH_PATTERN = TURKISH_APOSTROPHE_SUFFIX + "|" + TIKTOKEN_PATTERN


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_slice(lines: Iterable[str], destination: Path, byte_budget: int) -> dict:
    """Write complete UTF-8 records up to a byte quota; never cut a code point."""
    written = records = 0
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for text in lines:
            text = text.rstrip("\n")
            if not text:
                continue
            encoded = (text + "\n").encode("utf-8")
            if written + len(encoded) > byte_budget:
                break
            output.write(text + "\n")
            written += len(encoded)
            records += 1
    if written < byte_budget * 0.95:
        raise ValueError(
            f"{destination} supplied only {written:,} of {byte_budget:,} requested bytes"
        )
    return {"bytes": written, "records": records, "sha256": _sha256(destination)}


def _plain_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for raw in stream:
            repaired, _status = repair_text_encoding(raw.rstrip("\n"))
            if repaired is not None:
                yield repaired + "\n"


def _code_documents(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("language") == "code":
                # Blank-line separators prevent merges across unrelated files while
                # retaining indentation/newline statistics inside each source file.
                yield row["text"] + "\n"


def _synthetic_lines(source: Path, language: str) -> Iterable[str]:
    """Stream rendered synthetic chat records for one output language."""
    for shard in dataset_files(source):
        with shard.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("language") != language:
                    continue
                text = _sample_text(row)
                if text:
                    yield text + "\n"


def build_training_corpus(config: dict, output_dir: Path) -> tuple[list[Path], dict]:
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    budget = int(config["bytes_per_training_slice"])
    sources = config.get("training_sources")
    paths = {
        "en": corpus_dir / "en.txt",
        "tr": corpus_dir / "tr.txt",
    }
    if "synthetic_shards" in config:
        source = Path(config["synthetic_shards"])
        manifests = {
            language: _write_slice(_synthetic_lines(source, language), path, budget)
            for language, path in paths.items()
        }
        return list(paths.values()), manifests
    if sources is None:
        raise ValueError("config requires training_sources or synthetic_shards")
    paths["code"] = corpus_dir / "code.txt"
    manifests = {
        "en": _write_slice(_plain_lines(Path(sources["en"])), paths["en"], budget),
        "tr": _write_slice(_plain_lines(Path(sources["tr"])), paths["tr"], budget),
        "code": _write_slice(
            _code_documents(Path(sources["code_jsonl"])), paths["code"], budget
        ),
    }
    return list(paths.values()), manifests


def _turkish_weighted_corpus(corpus: list[Path]) -> list[Path]:
    """Return 25/50/25 EN/TR/code weighting by replaying the fixed TR slice.

    Every base slice has the same byte quota, so a second deterministic pass over
    tr.txt allocates half of tokenizer-training bytes to agglutinative Turkish
    without changing or duplicating language-model training examples.
    """
    by_name = {path.stem: path for path in corpus}
    return [by_name["en"], by_name["tr"], by_name["tr"], *(path for path in corpus if path.stem == "code")]


def train_byte_bpe(
    corpus: list[Path], vocab_size: int, destination: Path
) -> AmarkenTokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[0], byte_fallback=False))
    # No synthetic prefix: source starts, indentation, and prompt starts retain the
    # exact same byte representation. Regex splitting remains GPT-2 compatible.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train([str(path) for path in corpus], trainer)
    tokenizer.save(str(destination), pretty=True)
    return AmarkenTokenizer(destination, name=f"byte-bpe-{vocab_size // 1000}k")


def train_tiktoken_style_bpe(
    corpus: list[Path],
    vocab_size: int,
    destination: Path,
    *,
    pattern: str = TIKTOKEN_PATTERN,
    name: str | None = None,
) -> AmarkenTokenizer:
    """Train tiktoken-regex byte BPE using the production Rust trainer.

    Native tiktoken's public scratch trainer is educational and repeatedly scans
    Python lists for every merge. The tokenization semantics under test are the
    Unicode regex boundaries plus byte-level BPE, not that slow implementation.
    """
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[0], byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train([str(path) for path in corpus], trainer)
    tokenizer.save(str(destination), pretty=True)
    return AmarkenTokenizer(
        destination, name=name or f"tiktoken-style-bpe-{vocab_size // 1000}k"
    )


def _evaluation_texts(config: dict, corpus: list[Path]) -> dict[str, list[str]]:
    if "synthetic_shards" in config:
        corpus_by_name = {path.stem: path for path in corpus}
        limit = int(config.get("evaluation_documents_per_language", 2_000))
        if limit < 1:
            raise ValueError("evaluation_documents_per_language must be positive")
        result = {
            language: [
                line.rstrip("\n")
                for line in islice(
                    (line for line in _plain_lines(corpus_by_name[language]) if line.strip()),
                    limit,
                )
            ]
            for language in ("en", "tr")
        }
        result["code"] = [
            "def add(a, b):\n    return a + b\n",
            "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
            "class Cache:\n    def __init__(self):\n        self.values = {}\n",
        ]
        morphology = json.loads(Path(config["turkish_morphology"]).read_text(encoding="utf-8"))
        result["morphology"] = [form for family in morphology["families"] for form in family["forms"]]
        return result
    sources = config["evaluation_sources"]
    result: dict[str, list[str]] = {}
    for language in ("en", "tr"):
        result[language] = [
            line.rstrip("\n")
            for filename in sources[language]
            for line in _plain_lines(Path(filename))
            if line.strip()
        ]
    code = []
    for document in _code_documents(Path(sources["code_jsonl"])):
        # Fixed-size character windows keep a few huge validation files from
        # dominating while retaining real newlines and indentation.
        code.extend(
            document[start : start + 4096]
            for start in range(0, min(len(document), 16_384), 4096)
        )
    code.extend(
        [
            "def add(a, b):\n    return a + b\n",
            "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
            "class Cache:\n    def __init__(self):\n        self.values = {}\n",
        ]
    )
    result["code"] = [text for text in code if text]
    morphology = json.loads(
        Path(sources["turkish_morphology"]).read_text(encoding="utf-8")
    )
    result["morphology"] = [
        form for family in morphology["families"] for form in family["forms"]
    ]
    return result


def _token_metrics(adapter: AmarkenTokenizer, texts: list[str]) -> dict:
    tokens = words = characters = byte_like = unknown = whitespace = (
        roundtrip_failures
    ) = 0
    fallback_pieces: Counter[str] = Counter()
    unk_markers = {"<unk>", "[UNK]"}
    for text in texts:
        ids = adapter.encode(text)
        tokens += len(ids)
        words += len(text.split())
        characters += len(text)
        roundtrip_failures += adapter.decode(ids) != text
        for token_id in ids:
            piece = adapter.piece(token_id)
            unknown += piece in unk_markers
            byte_like += piece.startswith("<0x") and piece.endswith(">")
            if piece.startswith("<0x") and piece.endswith(">"):
                fallback_pieces[piece] += 1
            try:
                whitespace += (
                    bool(adapter.decode([token_id]))
                    and adapter.decode([token_id]).isspace()
                )
            except Exception:
                pass
    return {
        "documents": len(texts),
        "tokens": tokens,
        "tokens_per_word": tokens / max(words, 1),
        "tokens_per_character": tokens / max(characters, 1),
        "whitespace_token_fraction": whitespace / max(tokens, 1),
        "byte_fallback_fraction": byte_like / max(tokens, 1),
        "byte_fallback_pieces": dict(fallback_pieces.most_common(16)),
        "unknown_fraction": unknown / max(tokens, 1),
        "roundtrip_failures": roundtrip_failures,
    }


def _training_token_shares(
    adapter: AmarkenTokenizer, corpus: list[Path], turkish_weighted: bool = False
) -> dict[str, float]:
    def token_count(path: Path) -> int:
        total = 0
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            while chunk := handle.read(1024 * 1024):
                total += len(adapter.encode(chunk))
        return total

    counts = {
        path.stem: token_count(path)
        for path in corpus
    }
    if turkish_weighted:
        counts["tr"] *= 2
    total = sum(counts.values())
    return {name: count / total for name, count in counts.items()}


def evaluate(
    adapter: AmarkenTokenizer,
    texts: dict[str, list[str]],
    corpus: list[Path],
    *,
    require_code_qualification: bool = True,
) -> dict:
    slices = {
        name: _token_metrics(adapter, values)
        for name, values in texts.items()
        if name != "morphology"
    }
    morphology_counts = [len(adapter.encode(form)) for form in texts["morphology"]]
    indent = {}
    baseline_tokens = len(adapter.encode("\nreturn value\n"))
    for width in (1, 2, 4, 8):
        probe = "\n" + " " * width + "return value\n"
        indent[str(width)] = {
            "tokens": len(adapter.encode(probe)),
            # This directly measures indentation's context cost even when a
            # whitespace byte is merged into the following lexical token.
            "token_overhead_vs_unindented": len(adapter.encode(probe))
            - baseline_tokens,
            "roundtrip": adapter.decode(adapter.encode(probe)) == probe,
        }
    artifact_bytes = sum(path.stat().st_size for path in adapter.artifact_paths)
    failures = sum(metrics["roundtrip_failures"] for metrics in slices.values())
    requested_vocab_size = 16_000 if adapter.name == "byte-bpe-16k" else 12_000
    qualified = (
        adapter.vocab_size() == requested_vocab_size
        and failures == 0
        and all(metrics["unknown_fraction"] == 0 for metrics in slices.values())
        and (
            not require_code_qualification
            or (
                indent["4"]["roundtrip"]
                and indent["4"]["token_overhead_vs_unindented"] <= 1
                and indent["8"]["token_overhead_vs_unindented"] <= 1
            )
        )
    )
    return {
        "name": adapter.name,
        "vocab_size": adapter.vocab_size(),
        "artifact_bytes": artifact_bytes,
        "embedding_parameters_width_256": adapter.vocab_size() * 256,
        "embedding_parameters_width_512": adapter.vocab_size() * 512,
        "training_token_shares": _training_token_shares(
            adapter,
            corpus,
            turkish_weighted=adapter.name == "tiktoken-style-tr-weighted-bpe-12k",
        ),
        "slices": slices,
        "turkish_morphology": {
            "forms": len(morphology_counts),
            "mean_tokens": statistics.fmean(morphology_counts),
            "max_tokens": max(morphology_counts),
        },
        "indentation": indent,
        "qualified": qualified,
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in adapter.artifact_paths
        ],
    }


def run(config_path: Path, evaluate_only: bool = False) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format_version") != 1:
        raise ValueError("unsupported tokenizer-v2 config format")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus, corpus_manifest = build_training_corpus(config, output_dir)
    started = time.perf_counter()
    if evaluate_only:
        adapters: list[AmarkenTokenizer] = [
            AmarkenTokenizer(output_dir / "byte-bpe-12k.json", name="byte-bpe-12k"),
            AmarkenTokenizer(output_dir / "byte-bpe-16k.json", name="byte-bpe-16k"),
            (
                AmarkenTokenizer(
                    output_dir / "tiktoken-style-bpe-12k.json", name="tiktoken-style-bpe-12k"
                )
                if (output_dir / "tiktoken-style-bpe-12k.json").is_file()
                else train_tiktoken_style_bpe(
                    corpus, 12_000, output_dir / "tiktoken-style-bpe-12k.json"
                )
            ),
            (
                AmarkenTokenizer(
                    output_dir / "tiktoken-style-tr-bpe-12k.json",
                    name="tiktoken-style-apostrophe-bpe-12k",
                )
                if (output_dir / "tiktoken-style-tr-bpe-12k.json").is_file()
                else train_tiktoken_style_bpe(
                    corpus,
                    12_000,
                    output_dir / "tiktoken-style-tr-bpe-12k.json",
                    pattern=TIKTOKEN_TURKISH_PATTERN,
                    name="tiktoken-style-apostrophe-bpe-12k",
                )
            ),
            (
                AmarkenTokenizer(
                    output_dir / "tiktoken-style-tr-weighted-bpe-12k.json",
                    name="tiktoken-style-tr-weighted-bpe-12k",
                )
                if (output_dir / "tiktoken-style-tr-weighted-bpe-12k.json").is_file()
                else train_tiktoken_style_bpe(
                    _turkish_weighted_corpus(corpus),
                    12_000,
                    output_dir / "tiktoken-style-tr-weighted-bpe-12k.json",
                    name="tiktoken-style-tr-weighted-bpe-12k",
                )
            ),
        ]
    else:
        adapters = [
            train_byte_bpe(corpus, 12_000, output_dir / "byte-bpe-12k.json"),
            train_byte_bpe(corpus, 16_000, output_dir / "byte-bpe-16k.json"),
            train_tiktoken_style_bpe(
                corpus, 12_000, output_dir / "tiktoken-style-bpe-12k.json"
            ),
            train_tiktoken_style_bpe(
                corpus,
                12_000,
                output_dir / "tiktoken-style-tr-bpe-12k.json",
                pattern=TIKTOKEN_TURKISH_PATTERN,
                name="tiktoken-style-apostrophe-bpe-12k",
            ),
            train_tiktoken_style_bpe(
                _turkish_weighted_corpus(corpus),
                12_000,
                output_dir / "tiktoken-style-tr-weighted-bpe-12k.json",
                name="tiktoken-style-tr-weighted-bpe-12k",
            ),
        ]
    texts = _evaluation_texts(config, corpus)
    candidates = [
        evaluate(
            adapter,
            texts,
            corpus,
            require_code_qualification="synthetic_shards" not in config,
        )
        for adapter in adapters
    ]
    # Rank only qualified compact candidates under the fixed embedding budget.
    compact = [
        row for row in candidates if row["qualified"] and row["vocab_size"] <= 16_000
    ]
    winner = (
        min(
            compact,
            key=lambda row: (
                # Preserve embedding capacity first. Within one vocabulary size,
                # minimize balanced EN+TR word fertility plus code token density;
                # morphology and artifact bytes are deterministic tie breakers.
                row["vocab_size"],
                row["slices"]["en"]["tokens_per_word"]
                + row["slices"]["tr"]["tokens_per_word"]
                + row["slices"]["code"]["tokens_per_character"],
                row["turkish_morphology"]["mean_tokens"],
                row["artifact_bytes"],
            ),
        )
        if compact
        else None
    )
    # Tokenizer metrics expose real trade-offs rather than proving downstream LM
    # quality. Keep the apostrophe-aware production candidate and useful controls.
    probe_finalists = [
        name
        for name in (
            "tiktoken-style-apostrophe-bpe-12k",
            "tiktoken-style-tr-weighted-bpe-12k",
            "tiktoken-style-bpe-12k",
            "byte-bpe-16k",
        )
        if any(row["name"] == name and row["qualified"] for row in candidates)
    ]
    report = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "passed": winner is not None,
        "interpretation": "tokenizer qualification; no model capability claim",
        "config_sha256": _sha256(config_path),
        "training_corpus": corpus_manifest,
        "elapsed_seconds": time.perf_counter() - started,
        "recommended": winner["name"] if winner else None,
        "recommendation_scope": "metric-only provisional choice; model probe remains required",
        "model_probe_finalists": probe_finalists,
        "candidates": candidates,
    }
    report_path = Path(config["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, report_path)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--evaluate-only", action="store_true", help="reuse existing trained artifacts"
    )
    args = parser.parse_args()
    report = run(args.config, evaluate_only=args.evaluate_only)
    for row in report["candidates"]:
        print(
            row["name"],
            row["vocab_size"],
            row["qualified"],
            row["slices"]["code"]["whitespace_token_fraction"],
        )
    print("recommended", report["recommended"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
