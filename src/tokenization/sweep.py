"""Deterministic matched SentencePiece BPE sweep for English and Turkish.

All candidates share corpus, ordering, normalization, special tokens and trainer
settings; vocabulary size is the sole changed variable. Byte fallback reserves a
piece for every byte, making arbitrary UTF-8 encodable without an unknown token.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Iterable

import sentencepiece as spm


DEFAULT_ROOT = Path("data/raw/opus100/opus-100-corpus/v1.0/supervised/en-tr")
BYTE_PIECE = re.compile(r"^<0x[0-9A-F]{2}>$")
WORD = re.compile(r"\S+")
SPECIAL_PIECES = ("<unk>", "<s>", "</s>", "<pad>", "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>")


@dataclass(frozen=True)
class CorpusFiles:
    en_train: Path
    tr_train: Path
    en_eval: tuple[Path, ...]
    tr_eval: tuple[Path, ...]
    morphology: Path


@dataclass(frozen=True)
class LanguageMetrics:
    sentences: int
    words: int
    tokens: int
    fertility: float
    byte_tokens: int
    byte_fallback_rate: float
    unknown_tokens: int
    roundtrip_failures: int


@dataclass(frozen=True)
class MorphologyMetrics:
    words: int
    tokens: int
    pieces_per_word: float
    continuation_pieces_per_word: float
    inflected_to_lemma_piece_ratio: float
    exact_single_piece_rate: float
    family_details: tuple[dict, ...]


@dataclass(frozen=True)
class CandidateReport:
    vocabulary_size: int
    actual_vocabulary_size: int
    training_seconds: float
    model_bytes: int
    vocab_bytes: int
    artifact_bytes: int
    model_sha256: str
    english: LanguageMetrics
    turkish: LanguageMetrics
    morphology: MorphologyMetrics
    stress_byte_fallback_rate: float
    stress_roundtrip_failures: int


@dataclass(frozen=True)
class SweepReport:
    passed: bool
    corpus: str
    corpus_archive_sha256: str | None
    sentencepiece_version: str
    trainer: dict
    corpus_files: dict[str, dict]
    candidates: tuple[CandidateReport, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def default_corpus_files(morphology: Path = Path("data/tokenizer_eval/tr_morphology.json")) -> CorpusFiles:
    return CorpusFiles(
        en_train=DEFAULT_ROOT / "opus.en-tr-train.en",
        tr_train=DEFAULT_ROOT / "opus.en-tr-train.tr",
        en_eval=(DEFAULT_ROOT / "opus.en-tr-dev.en", DEFAULT_ROOT / "opus.en-tr-test.en"),
        tr_eval=(DEFAULT_ROOT / "opus.en-tr-dev.tr", DEFAULT_ROOT / "opus.en-tr-test.tr"),
        morphology=morphology,
    )


def _validate_corpus(files: CorpusFiles) -> dict[str, dict]:
    paths = (files.en_train, files.tr_train, *files.en_eval, *files.tr_eval, files.morphology)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing tokenizer inputs: {missing}")
    en_train_lines, tr_train_lines = _line_count(files.en_train), _line_count(files.tr_train)
    if en_train_lines != tr_train_lines:
        raise ValueError("English and Turkish training files are not aligned")
    for en_path, tr_path in zip(files.en_eval, files.tr_eval):
        if _line_count(en_path) != _line_count(tr_path):
            raise ValueError(f"evaluation files are not aligned: {en_path}, {tr_path}")
    return {
        str(path): {"bytes": path.stat().st_size, "lines": _line_count(path), "sha256": _sha256(path)}
        for path in paths
    }


def _train(files: CorpusFiles, vocabulary_size: int, prefix: Path) -> float:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spm.SentencePieceTrainer.train(
        input=f"{files.en_train},{files.tr_train}",
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=vocabulary_size,
        character_coverage=1.0,
        byte_fallback=True,
        hard_vocab_limit=True,
        # Identity normalization preserves Turkish casing/diacritics and permits
        # strict decode(encode(text)) checks rather than normalized equivalence.
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=True,
        split_digits=True,
        split_by_whitespace=True,
        # OPUS-100 contains an approximately 9k-character outlier. A 16k cap
        # keeps it in every candidate while still rejecting pathological input.
        max_sentence_length=16384,
        num_threads=1,
        minloglevel=1,
        shuffle_input_sentence=False,
        input_sentence_size=0,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=list(SPECIAL_PIECES[4:]),
    )
    return time.perf_counter() - started


def _texts(paths: Iterable[Path]) -> Iterable[str]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            for line in stream:
                # Only record delimiters are removed; content whitespace is retained
                # and therefore participates in exact round-trip verification.
                yield line.rstrip("\r\n")


def _language_metrics(processor: spm.SentencePieceProcessor, paths: tuple[Path, ...]) -> LanguageMetrics:
    sentences = words = tokens = byte_tokens = unknown_tokens = failures = 0
    unk_id = processor.unk_id()
    for text in _texts(paths):
        ids = processor.encode(text, out_type=int)
        pieces = [processor.id_to_piece(identifier) for identifier in ids]
        sentences += 1
        words += len(WORD.findall(text))
        tokens += len(ids)
        byte_tokens += sum(bool(BYTE_PIECE.match(piece)) for piece in pieces)
        unknown_tokens += sum(identifier == unk_id for identifier in ids)
        failures += processor.decode(ids) != text
    return LanguageMetrics(
        sentences=sentences,
        words=words,
        tokens=tokens,
        fertility=tokens / words if words else 0.0,
        byte_tokens=byte_tokens,
        byte_fallback_rate=byte_tokens / tokens if tokens else 0.0,
        unknown_tokens=unknown_tokens,
        roundtrip_failures=failures,
    )


def _morphology_metrics(processor: spm.SentencePieceProcessor, path: Path) -> MorphologyMetrics:
    families = json.loads(path.read_text(encoding="utf-8"))["families"]
    total_words = total_tokens = single_piece = 0
    ratios: list[float] = []
    details: list[dict] = []
    for family in families:
        lemma = family["lemma"]
        forms = family["forms"]
        lemma_count = len(processor.encode(lemma, out_type=int))
        encoded = []
        for word in forms:
            pieces = processor.encode(word, out_type=str)
            total_words += 1
            total_tokens += len(pieces)
            single_piece += len(pieces) == 1
            # Exclude the lemma itself: this aggregate is specifically the
            # extra segmentation introduced by Turkish inflectional suffixes.
            if word != lemma:
                ratios.append(len(pieces) / lemma_count)
            encoded.append({"word": word, "pieces": pieces, "piece_count": len(pieces)})
        details.append({"lemma": lemma, "lemma_piece_count": lemma_count, "forms": encoded})
    return MorphologyMetrics(
        words=total_words,
        tokens=total_tokens,
        pieces_per_word=total_tokens / total_words,
        continuation_pieces_per_word=(total_tokens - total_words) / total_words,
        inflected_to_lemma_piece_ratio=sum(ratios) / len(ratios),
        exact_single_piece_rate=single_piece / total_words,
        family_details=tuple(details),
    )


STRESS_TEXTS = (
    "Türkçe: Iğdır, İstanbul, Çeşme; ıİğĞşŞçÇöÖüÜ.",
    "emoji 🧠✨, math ∀x∈ℝ, code `x += 1`, العربية, 中文, हिन्दी",
    "spaces  stay   exact\twith a tab",
    "combining: e\u0301 vs é; rare: 𐱅𐰇𐰼𐰚",
)


def _stress_metrics(processor: spm.SentencePieceProcessor) -> tuple[float, int]:
    token_count = byte_count = failures = 0
    for text in STRESS_TEXTS:
        ids = processor.encode(text, out_type=int)
        token_count += len(ids)
        byte_count += sum(bool(BYTE_PIECE.match(processor.id_to_piece(identifier))) for identifier in ids)
        failures += processor.decode(ids) != text
    return byte_count / token_count, failures


def train_sweep(
    files: CorpusFiles,
    output_dir: Path,
    vocabulary_sizes: tuple[int, ...] = (8_000, 12_000, 16_000),
    archive: Path | None = Path("data/raw/opus100/en-tr-v1.0.tar.gz"),
) -> SweepReport:
    """Train and evaluate matched candidates; existing artifacts are overwritten."""
    if len(set(vocabulary_sizes)) != len(vocabulary_sizes) or any(size < 512 for size in vocabulary_sizes):
        raise ValueError("vocabulary sizes must be unique and at least 512 with byte fallback")
    manifest = _validate_corpus(files)
    candidates = []
    for size in vocabulary_sizes:
        prefix = output_dir / f"amarken-en-tr-{size // 1000}k"
        training_seconds = _train(files, size, prefix)
        model_path, vocab_path = prefix.with_suffix(".model"), prefix.with_suffix(".vocab")
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
        english = _language_metrics(processor, files.en_eval)
        turkish = _language_metrics(processor, files.tr_eval)
        morphology = _morphology_metrics(processor, files.morphology)
        stress_rate, stress_failures = _stress_metrics(processor)
        candidates.append(
            CandidateReport(
                vocabulary_size=size,
                actual_vocabulary_size=processor.vocab_size(),
                training_seconds=training_seconds,
                model_bytes=model_path.stat().st_size,
                vocab_bytes=vocab_path.stat().st_size,
                artifact_bytes=model_path.stat().st_size + vocab_path.stat().st_size,
                model_sha256=_sha256(model_path),
                english=english,
                turkish=turkish,
                morphology=morphology,
                stress_byte_fallback_rate=stress_rate,
                stress_roundtrip_failures=stress_failures,
            )
        )
    passed = all(
        candidate.actual_vocabulary_size == candidate.vocabulary_size
        and candidate.english.unknown_tokens == candidate.turkish.unknown_tokens == 0
        and candidate.english.roundtrip_failures == candidate.turkish.roundtrip_failures == 0
        and candidate.stress_roundtrip_failures == 0
        for candidate in candidates
    )
    return SweepReport(
        passed=passed,
        corpus="OPUS-100 v1.0 supervised en-tr",
        corpus_archive_sha256=_sha256(archive) if archive is not None and archive.is_file() else None,
        sentencepiece_version=spm.__version__,
        trainer={
            "model_type": "bpe",
            "byte_fallback": True,
            "character_coverage": 1.0,
            "normalization_rule_name": "identity",
            "hard_vocab_limit": True,
            "num_threads": 1,
            "max_sentence_length": 16384,
            "shuffle_input_sentence": False,
            "special_pieces": list(SPECIAL_PIECES),
        },
        corpus_files=manifest,
        candidates=tuple(candidates),
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizers"))
    parser.add_argument("--report", type=Path, default=Path("experiments/tokenizer_sweep.json"))
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[8_000, 12_000, 16_000])
    args = parser.parse_args()
    result = train_sweep(default_corpus_files(), args.output_dir, tuple(args.vocab_sizes))
    text = json.dumps(result.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)
    # The detailed morphology traces make the JSON intentionally large; keep
    # stdout operationally useful and leave full provenance in --report.
    print(f"tokenizer sweep: {'PASS' if result.passed else 'FAIL'}; report={args.report}")
    for candidate in result.candidates:
        print(
            f"{candidate.vocabulary_size}: EN fertility={candidate.english.fertility:.4f}, "
            f"TR fertility={candidate.turkish.fertility:.4f}, "
            f"TR morphology={candidate.morphology.pieces_per_word:.4f}, "
            f"artifact={candidate.artifact_bytes} bytes"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(args.report)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
