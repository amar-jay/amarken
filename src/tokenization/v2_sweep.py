"""Train and gate balanced EN/TR/code tokenizer candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Iterable, Protocol

import sentencepiece as spm
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


SPECIAL_TOKENS = (
    "<unk>", "<s>", "</s>", "<pad>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|code|>",
)


class Adapter(Protocol):
    name: str
    artifact_paths: tuple[Path, ...]

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def piece(self, token_id: int) -> str: ...
    def vocab_size(self) -> int: ...


class HFAdapter:
    def __init__(self, name: str, path: Path):
        self.name, self.path = name, path
        if path.is_dir():
            vocab, merges = path / "vocab.json", path / "merges.txt"
            self.tokenizer = Tokenizer(models.BPE.from_file(str(vocab), str(merges), unk_token="<|endoftext|>"))
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
            self.tokenizer.decoder = decoders.ByteLevel()
            self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
            self.artifact_paths = tuple(
                candidate for candidate in (
                    vocab, merges, path / "tokenizer_config.json", path / "special_tokens_map.json"
                ) if candidate.is_file()
            )
        else:
            self.tokenizer = Tokenizer.from_file(str(path))
            self.artifact_paths = (path,)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def piece(self, token_id: int) -> str:
        return self.tokenizer.id_to_token(token_id) or ""

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)


class SPAdapter:
    def __init__(self, name: str, model_path: Path):
        self.name = name
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))
        self.artifact_paths = (model_path, model_path.with_suffix(".vocab"))

    def encode(self, text: str) -> list[int]:
        return self.processor.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.processor.decode(ids)

    def piece(self, token_id: int) -> str:
        return self.processor.id_to_piece(token_id)

    def vocab_size(self) -> int:
        return self.processor.vocab_size()


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
        raise ValueError(f"{destination} supplied only {written:,} of {byte_budget:,} requested bytes")
    return {"bytes": written, "records": records, "sha256": _sha256(destination)}


def _plain_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        yield from stream


def _code_documents(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("language") == "code":
                # Blank-line separators prevent merges across unrelated files while
                # retaining indentation/newline statistics inside each source file.
                yield row["text"] + "\n"


def build_training_corpus(config: dict, output_dir: Path) -> tuple[list[Path], dict]:
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    budget = int(config["bytes_per_training_slice"])
    sources = config["training_sources"]
    paths = {
        "en": corpus_dir / "en.txt",
        "tr": corpus_dir / "tr.txt",
        "code": corpus_dir / "code.txt",
    }
    manifests = {
        "en": _write_slice(_plain_lines(Path(sources["en"])), paths["en"], budget),
        "tr": _write_slice(_plain_lines(Path(sources["tr"])), paths["tr"], budget),
        "code": _write_slice(_code_documents(Path(sources["code_jsonl"])), paths["code"], budget),
    }
    return list(paths.values()), manifests


def train_byte_bpe(corpus: list[Path], vocab_size: int, destination: Path) -> HFAdapter:
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[0], byte_fallback=False))
    # No synthetic prefix: source starts, indentation, and prompt starts retain the
    # exact same byte representation. Regex splitting remains GPT-2 compatible.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
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
    return HFAdapter(f"byte-bpe-{vocab_size // 1000}k", destination)


def train_sentencepiece(corpus: list[Path], model_type: str, destination_prefix: Path) -> SPAdapter:
    # The corrected candidate preserves bytes/whitespace, learns whitespace-only
    # pieces, and sees the identical balanced corpus as byte-BPE.
    spm.SentencePieceTrainer.train(
        input=",".join(str(path) for path in corpus),
        model_prefix=str(destination_prefix),
        model_type=model_type,
        vocab_size=12_000,
        character_coverage=1.0,
        byte_fallback=True,
        hard_vocab_limit=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=False,
        split_by_whitespace=False,
        allow_whitespace_only_pieces=True,
        split_digits=True,
        max_sentence_length=65_536,
        num_threads=1,
        shuffle_input_sentence=False,
        input_sentence_size=0,
        minloglevel=1,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        user_defined_symbols=list(SPECIAL_TOKENS[4:]),
    )
    return SPAdapter(f"sp-{model_type}-12k", destination_prefix.with_suffix(".model"))


def _evaluation_texts(config: dict) -> dict[str, list[str]]:
    sources = config["evaluation_sources"]
    result: dict[str, list[str]] = {}
    for language in ("en", "tr"):
        result[language] = [
            line.rstrip("\n")
            for filename in sources[language]
            for line in Path(filename).open("r", encoding="utf-8", errors="strict")
            if line.strip()
        ]
    code = []
    for document in _code_documents(Path(sources["code_jsonl"])):
        # Fixed-size character windows keep a few huge validation files from
        # dominating while retaining real newlines and indentation.
        code.extend(document[start:start + 4096] for start in range(0, min(len(document), 16_384), 4096))
    code.extend([
        "def add(a, b):\n    return a + b\n",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
        "class Cache:\n    def __init__(self):\n        self.values = {}\n",
    ])
    result["code"] = [text for text in code if text]
    morphology = json.loads(Path(sources["turkish_morphology"]).read_text(encoding="utf-8"))
    result["morphology"] = [form for family in morphology["families"] for form in family["forms"]]
    return result


def _token_metrics(adapter: Adapter, texts: list[str]) -> dict:
    tokens = words = characters = byte_like = unknown = whitespace = roundtrip_failures = 0
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
                whitespace += bool(adapter.decode([token_id])) and adapter.decode([token_id]).isspace()
            except Exception:
                pass
    return {
        "documents": len(texts), "tokens": tokens,
        "tokens_per_word": tokens / max(words, 1),
        "tokens_per_character": tokens / max(characters, 1),
        "whitespace_token_fraction": whitespace / max(tokens, 1),
        "byte_fallback_fraction": byte_like / max(tokens, 1),
        "byte_fallback_pieces": dict(fallback_pieces.most_common(16)),
        "unknown_fraction": unknown / max(tokens, 1),
        "roundtrip_failures": roundtrip_failures,
    }


def _training_token_shares(adapter: Adapter, corpus: list[Path]) -> dict[str, float]:
    counts = {
        path.stem: len(adapter.encode(path.read_text(encoding="utf-8")))
        for path in corpus
    }
    total = sum(counts.values())
    return {name: count / total for name, count in counts.items()}


def evaluate(adapter: Adapter, texts: dict[str, list[str]], corpus: list[Path]) -> dict:
    slices = {name: _token_metrics(adapter, values) for name, values in texts.items() if name != "morphology"}
    morphology_counts = [len(adapter.encode(form)) for form in texts["morphology"]]
    indent = {}
    baseline_tokens = len(adapter.encode("\nreturn value\n"))
    for width in (1, 2, 4, 8):
        probe = "\n" + " " * width + "return value\n"
        indent[str(width)] = {
            "tokens": len(adapter.encode(probe)),
            # This directly measures indentation's context cost even when a
            # whitespace byte is merged into the following lexical token.
            "token_overhead_vs_unindented": len(adapter.encode(probe)) - baseline_tokens,
            "roundtrip": adapter.decode(adapter.encode(probe)) == probe,
        }
    artifact_bytes = sum(path.stat().st_size for path in adapter.artifact_paths)
    failures = sum(metrics["roundtrip_failures"] for metrics in slices.values())
    qualified = (
        failures == 0
        and all(metrics["unknown_fraction"] == 0 for metrics in slices.values())
        and indent["4"]["roundtrip"]
        and indent["4"]["token_overhead_vs_unindented"] <= 1
        and indent["8"]["token_overhead_vs_unindented"] <= 1
    )
    return {
        "name": adapter.name, "vocab_size": adapter.vocab_size(),
        "artifact_bytes": artifact_bytes,
        "embedding_parameters_width_256": adapter.vocab_size() * 256,
        "embedding_parameters_width_512": adapter.vocab_size() * 512,
        "training_token_shares": _training_token_shares(adapter, corpus),
        "slices": slices,
        "turkish_morphology": {
            "forms": len(morphology_counts),
            "mean_tokens": statistics.fmean(morphology_counts),
            "max_tokens": max(morphology_counts),
        },
        "indentation": indent,
        "qualified": qualified,
        "artifacts": [{"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in adapter.artifact_paths],
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
        adapters: list[Adapter] = [
            HFAdapter("byte-bpe-12k", output_dir / "byte-bpe-12k.json"),
            HFAdapter("byte-bpe-16k", output_dir / "byte-bpe-16k.json"),
            SPAdapter("sp-bpe-12k", output_dir / "sp-bpe-12k.model"),
            SPAdapter("sp-unigram-12k", output_dir / "sp-unigram-12k.model"),
            HFAdapter(config["external"]["name"], Path(config["external"]["tokenizer_path"])),
        ]
    else:
        adapters = [
            train_byte_bpe(corpus, 12_000, output_dir / "byte-bpe-12k.json"),
            train_byte_bpe(corpus, 16_000, output_dir / "byte-bpe-16k.json"),
            train_sentencepiece(corpus, "bpe", output_dir / "sp-bpe-12k"),
            train_sentencepiece(corpus, "unigram", output_dir / "sp-unigram-12k"),
            HFAdapter(config["external"]["name"], Path(config["external"]["tokenizer_path"])),
        ]
    texts = _evaluation_texts(config)
    candidates = [evaluate(adapter, texts, corpus) for adapter in adapters]
    # Rank only qualified compact candidates; the external 49k tokenizer remains
    # a quality reference because its embedding cost changes the model budget.
    compact = [row for row in candidates if row["qualified"] and row["vocab_size"] <= 16_000]
    winner = min(
        compact,
        key=lambda row: (
            # Preserve embedding capacity first. Within one vocabulary size,
            # minimize balanced EN+TR word fertility plus code token density;
            # morphology and artifact bytes are deterministic tie breakers.
            row["vocab_size"],
            row["slices"]["en"]["tokens_per_word"]
            + row["slices"]["tr"]["tokens_per_word"]
            + row["slices"]["code"]["tokens_per_character"],
            row["turkish_morphology"]["mean_tokens"], row["artifact_bytes"],
        ),
    ) if compact else None
    # Tokenizer metrics expose real trade-offs rather than proving downstream LM
    # quality. These three cover the strongest fixed-12k code/morphology option,
    # the balanced 12k unigram option, and the higher-capacity byte-BPE endpoint.
    probe_finalists = [
        name for name in ("sp-bpe-12k", "sp-unigram-12k", "byte-bpe-16k")
        if any(row["name"] == name and row["qualified"] for row in candidates)
    ]
    report = {
        "format_version": 1, "experiment_id": config["experiment_id"],
        "passed": winner is not None,
        "interpretation": "tokenizer qualification; no model capability claim",
        "config_sha256": _sha256(config_path),
        "training_corpus": corpus_manifest,
        "external_provenance": config["external"],
        "elapsed_seconds": time.perf_counter() - started,
        "recommended": winner["name"] if winner else None,
        "recommendation_scope": "metric-only provisional choice; model probe remains required",
        "model_probe_finalists": probe_finalists,
        "candidates": candidates,
    }
    report_path = Path(config["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, report_path)
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tokenizer_v2.json"))
    parser.add_argument("--evaluate-only", action="store_true", help="reuse existing trained artifacts")
    args = parser.parse_args()
    report = run(args.config, evaluate_only=args.evaluate_only)
    for row in report["candidates"]:
        print(row["name"], row["vocab_size"], row["qualified"], row["slices"]["code"]["whitespace_token_fraction"])
    print("recommended", report["recommended"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
