"""Train and gate balanced tokenizer."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time
from typing import Iterable
from itertools import islice

from tabulate import tabulate
from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
)

from src.tokenization.tokenizer import AmarkenTokenizer, SPECIAL_TOKENS
from src.utils.files import sha256_file, write_report
from src.tokenization.text import repair_text_encoding
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
    return {"bytes": written, "records": records, "sha256": sha256_file(destination)}


def _plain_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for raw in stream:
            repaired, _status = repair_text_encoding(raw.rstrip("\n"))
            if repaired is not None:
                yield repaired + "\n"


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
    manifests = {
        "en": _write_slice(_plain_lines(Path(sources["en"])), paths["en"], budget),
        "tr": _write_slice(_plain_lines(Path(sources["tr"])), paths["tr"], budget),
    }
    return list(paths.values()), manifests


def _turkish_weighted_corpus(corpus: list[Path]) -> list[Path]:
    """Return 1/2 EN and 1/2 TR weighting by replaying the fixed TR slice.

    Every base slice has the same byte quota, so a second deterministic pass over
    tr.txt gives agglutinative Turkish twice the tokenizer-training weight of
    English without changing the underlying source slices.
    """
    by_name = {path.stem: path for path in corpus}
    return [by_name["en"], by_name["tr"], by_name["tr"]]


def train_byte_bpe(
    corpus: list[Path], vocab_size: int, destination: Path, *, min_frequency: int = 2
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
        min_frequency=min_frequency,
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
    min_frequency: int = 2,
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
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train([str(path) for path in corpus], trainer)
    tokenizer.save(str(destination), pretty=True)
    return AmarkenTokenizer(
        destination, name=name or f"tiktoken-bpe-{vocab_size // 1000}k"
    )


def _evaluation_texts(config: dict, corpus: list[Path]) -> dict[str, list[str]]:
    def morphology_forms(path: str | Path) -> list[str]:
        morphology = json.loads(Path(path).read_text(encoding="utf-8"))
        return [form for family in morphology["families"] for form in family["forms"]]

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
        result["morphology_en"] = morphology_forms(config["english_morphology"])
        result["morphology_tr"] = morphology_forms(config["turkish_morphology"])
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
    result["morphology_en"] = morphology_forms(sources["english_morphology"])
    result["morphology_tr"] = morphology_forms(sources["turkish_morphology"])
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
) -> dict:
    slices = {
        name: _token_metrics(adapter, values)
        for name, values in texts.items()
        if not name.startswith("morphology_")
    }
    morphology_en_counts = [len(adapter.encode(form)) for form in texts["morphology_en"]]
    morphology_tr_counts = [len(adapter.encode(form)) for form in texts["morphology_tr"]]
    artifact_bytes = sum(path.stat().st_size for path in adapter.artifact_paths)
    failures = sum(metrics["roundtrip_failures"] for metrics in slices.values())
    requested_vocab_size = 16_000 if adapter.name == "byte-bpe-16k" else 12_000
    qualified = (
        adapter.vocab_size() == requested_vocab_size
        and failures == 0
        and all(metrics["unknown_fraction"] == 0 for metrics in slices.values())
    )
    return {
        "name": adapter.name,
        "artifact_bytes": artifact_bytes,
        "training_token_shares": _training_token_shares(
            adapter,
            corpus,
            turkish_weighted=adapter.name == "tiktoken-tr-weighted-bpe-12k",
        ),
        "slices": slices,
        "english_morphology": {
            "forms": len(morphology_en_counts),
            "mean_tokens": statistics.fmean(morphology_en_counts),
            "max_tokens": max(morphology_en_counts),
        },
        "turkish_morphology": {
            "forms": len(morphology_tr_counts),
            "mean_tokens": statistics.fmean(morphology_tr_counts),
            "max_tokens": max(morphology_tr_counts),
        },
        "qualified": qualified,
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in adapter.artifact_paths
        ],
    }


def run(config: dict, evaluate_only: bool = False) -> dict:

    output_dir = Path(config["output_dir"])
    min_frequency = config.get("min_frequency", 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus, corpus_manifest = build_training_corpus(config, output_dir)
    started = time.perf_counter()
    if evaluate_only:
        adapters: list[AmarkenTokenizer] = [
            AmarkenTokenizer(output_dir / "byte-bpe-12k.json", name="byte-bpe-12k"),
            AmarkenTokenizer(output_dir / "byte-bpe-16k.json", name="byte-bpe-16k"),
            (
                AmarkenTokenizer(
                    output_dir / "tiktoken-bpe-12k.json", name="tiktoken-bpe-12k"
                )
                if (output_dir / "tiktoken-bpe-12k.json").is_file()
                else train_tiktoken_style_bpe(
                    corpus, 12_000, output_dir / "tiktoken-bpe-12k.json"
                )
            ),
            (
                AmarkenTokenizer(
                    output_dir / "tiktoken-tr-bpe-12k.json",
                    name="tiktoken-tr-bpe-12k",
                )
                if (output_dir / "tiktoken-tr-bpe-12k.json").is_file()
                else train_tiktoken_style_bpe(
                    corpus,
                    12_000,
                    output_dir / "tiktoken-tr-bpe-12k.json",
                    pattern=TIKTOKEN_TURKISH_PATTERN,
                    name="tiktoken-tr-bpe-12k",
                )
            ),
            (
                AmarkenTokenizer(
                    output_dir / "tiktoken-tr-weighted-bpe-12k.json",
                    name="tiktoken-tr-weighted-bpe-12k",
                )
                if (output_dir / "tiktoken-tr-weighted-bpe-12k.json").is_file()
                else train_tiktoken_style_bpe(
                    _turkish_weighted_corpus(corpus),
                    12_000,
                    output_dir / "tiktoken-tr-weighted-bpe-12k.json",
                    name="tiktoken-tr-weighted-bpe-12k",
                )
            ),
        ]
    else:
        adapters = [
            train_byte_bpe(corpus, 12_000, output_dir / "byte-bpe-12k.json", min_frequency=min_frequency),
            train_byte_bpe(corpus, 16_000, output_dir / "byte-bpe-16k.json", min_frequency=min_frequency),
            train_tiktoken_style_bpe(
                corpus, 12_000, output_dir / "tiktoken-bpe-12k.json", min_frequency=min_frequency
            ),
            train_tiktoken_style_bpe(
                corpus,
                12_000,
                output_dir / "tiktoken-tr-bpe-12k.json",
                pattern=TIKTOKEN_TURKISH_PATTERN,
                name="tiktoken-tr-bpe-12k", min_frequency=min_frequency
            ),
            train_tiktoken_style_bpe(
                _turkish_weighted_corpus(corpus),
                12_000,
                output_dir / "tiktoken-tr-weighted-bpe-12k.json",
                name="tiktoken-tr-weighted-bpe-12k", min_frequency=min_frequency
            ),
        ]
    texts = _evaluation_texts(config, corpus)
    candidates = [evaluate(adapter, texts, corpus) for adapter in adapters]
    probe_finalists = [
        name
        for name in (
            "tiktoken-tr-bpe-12k",
            "tiktoken-tr-weighted-bpe-12k",
            "tiktoken-bpe-12k",
            "byte-bpe-16k",
        )
        if any(row["name"] == name and row["qualified"] for row in candidates)
    ]
    report = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "interpretation": "tokenizer qualification; no model capability claim",
        "training_corpus": corpus_manifest,
        "elapsed_seconds": time.perf_counter() - started,
        "model_probe_finalists": probe_finalists,
        "candidates": candidates,
    }
    return report

def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--evaluate-only", action="store_true", help="reuse existing trained artifacts"
    )
    parser.add_argument(
        "--report", action="store_true", help="write the JSON report"
    )

    args = parser.parse_args()

    config_path = args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format_version") != 1:
        raise ValueError("unsupported tokenizer-v2 config format")
    report = run(config, evaluate_only=args.evaluate_only)
    headers = [
        "Tokenizer",
        "EN Train Share",
        "TR Train Share",
        "EN tok/word",
        "TR tok/word",
        "EN tok/char",
        "TR tok/char",
        "EN Morph Mean",
        "EN Morph Max",
        "TR Morph Mean",
        "TR Morph Max",
    ]
    table = []
    for row in report["candidates"]:
        en = row["slices"]["en"]
        tr = row["slices"]["tr"]
        shares = row["training_token_shares"]
        table.append(
            [
                row["name"],
                f'{shares["en"]:.2%}',
                f'{shares["tr"]:.2%}',
                f'{en["tokens_per_word"]:.4f}',
                f'{tr["tokens_per_word"]:.4f}',
                f'{en["tokens_per_character"]:.4f}',
                f'{tr["tokens_per_character"]:.4f}',
                f'{row["english_morphology"]["mean_tokens"]:.4f}',
                row["english_morphology"]["max_tokens"],
                f'{row["turkish_morphology"]["mean_tokens"]:.4f}',
                row["turkish_morphology"]["max_tokens"],
            ]
        )
    print(tabulate(table, headers=headers, tablefmt="github"))
    if args.report or not args.evaluate_only:
        report["config_sha256"] = sha256_file(config_path)
        report_path = Path(config["report"])
        write_report(report, report_path)
        print(f"Wrote tokenizer sweep report to {report_path} ({len(report['candidates'])} candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
