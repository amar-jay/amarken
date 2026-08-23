"""Color token boundaries for deterministic random samples from JSONL datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys

from src.data.proxy import repair_text_encoding

from .tokenizer import AmarkenTokenizer

# Dark 256-color backgrounds with white foreground remain distinguishable in
# common light/dark terminals. Position-based rotation keeps repeated token IDs
# visually separate, which is essential for diagnosing whitespace loops.
BACKGROUND_COLORS = (17, 22, 52, 53, 58, 88, 89, 94, 95, 100, 23, 24, 25, 54, 55, 60)
RESET = "\x1b[0m"


@dataclass(frozen=True)
class Sample:
    document_id: str
    language: str
    domain: str
    source_id: str
    text: str


def _visible(text: str) -> str:
    """Expose layout characters without losing their line structure."""
    return (
        text.replace(" ", "·")
        .replace("\t", "→\t")
        .replace("\r", "␍")
        .replace("\n", "↵\n")
    )


def _token_surface(adapter: AmarkenTokenizer, token_id: int) -> str:
    # Like tiktoken's educational visualizer, tolerate tokens that contain only
    # a fragment of one UTF-8 code point. The piece legend retains raw identity.
    try:
        surface = adapter.decode([token_id])
    except (UnicodeDecodeError, ValueError):
        surface = "�"
    return _visible(surface or "�")


def probable_mojibake(text: str) -> bool:
    """Use the corpus repair policy, avoiding false alarms on Turkish â."""
    _repaired, status = repair_text_encoding(text)
    return status != "unchanged"


def _ids_and_offsets(
    adapter: AmarkenTokenizer, text: str
) -> tuple[list[int], list[tuple[int, int]]]:
    return adapter.encode_with_offsets(text)


def render_tokens(
    adapter: AmarkenTokenizer, text: str, color: bool = True
) -> tuple[str, list[dict]]:
    ids, offsets = _ids_and_offsets(adapter, text)
    blocks = []
    legend = []
    for position, (token_id, offset) in enumerate(zip(ids, offsets)):
        surface = _token_surface(adapter, token_id)
        legend.append(
            {
                "position": position,
                "id": token_id,
                "piece": adapter.piece(token_id),
                "surface": surface,
                "offset": offset,
            }
        )

    # ANSI escapes cannot be inserted between bytes of one Unicode code point.
    # Group tokens whose character offsets overlap (emoji/non-ASCII byte splits),
    # then color the exact source slice. The legend still exposes every raw token.
    groups: list[tuple[int, int, int]] = []
    for position, (start, end) in enumerate(offsets):
        if groups and (
            start < groups[-1][2] or (groups[-1][1] == groups[-1][2] == start)
        ):
            first, group_start, group_end = groups[-1]
            groups[-1] = (first, min(group_start, start), max(group_end, end))
        else:
            groups.append((position, start, end))
    cursor = 0
    for position, start, end in groups:
        if start > cursor:
            blocks.append(_visible(text[cursor:start]))
        surface = _visible(text[start:end])
        if color:
            background = BACKGROUND_COLORS[position % len(BACKGROUND_COLORS)]
            blocks.append(f"\x1b[38;5;15;48;5;{background}m{surface}{RESET}")
        else:
            blocks.append(f"⟦{surface}⟧")
        cursor = max(cursor, end)
    if cursor < len(text):
        blocks.append(_visible(text[cursor:]))
    return "".join(blocks), legend


def _crop(text: str, maximum: int, rng: random.Random) -> str:
    if len(text) <= maximum:
        return text
    start = rng.randrange(0, len(text) - maximum + 1)
    # Prefer complete lines when nearby; prose without newlines uses exact windows.
    next_line = text.find("\n", start, min(len(text), start + 200))
    if next_line != -1:
        start = next_line + 1
    end = min(len(text), start + maximum)
    next_end = text.find("\n", end, min(len(text), end + 200))
    if next_end != -1:
        end = next_end + 1
    return text[start:end]


def dataset_files(dataset: Path) -> list[Path]:
    """Resolve one JSONL file or a synthetic shard directory in stable order."""
    if dataset.is_file():
        return [dataset]
    if not dataset.is_dir():
        raise ValueError(f"dataset does not exist: {dataset}")
    paths = sorted(dataset.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"dataset directory has no *.jsonl files: {dataset}")
    return paths


def _sample_text(row: dict) -> str | None:
    if row.get("text"):
        return str(row["text"])
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    role_tags = {
        "system": "<|system|>",
        "developer": "<|developer|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
        "tool": "<|tool|>",
    }
    rendered = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if content is None:
            continue
        content = str(content).strip()
        if not content:
            continue
        role = str(message.get("role", "unknown")).lower()
        tag = role_tags.get(role, f"<|{role}|>")
        rendered.append(f"{tag}: {content} \n<|end|>")
    if not rendered:
        return None
    return "\n".join(rendered) if rendered else None


def reservoir_samples(
    dataset: Path,
    count: int,
    seed: int,
    maximum_characters: int,
    language: str | None = None,
    domain: str | None = None,
) -> list[Sample]:
    """Uniformly sample eligible documents in one streaming pass."""
    if count < 1 or maximum_characters < 1:
        raise ValueError("count and maximum_characters must be positive")
    rng = random.Random(seed)
    reservoir: list[dict] = []
    eligible = 0
    for path in dataset_files(dataset):
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            for line in stream:
                row = json.loads(line)
                if language is not None and row.get("language") != language:
                    continue
                row_domain = row.get("category", row.get("domain"))
                if domain is not None and row_domain != domain:
                    continue
                text = _sample_text(row)
                if not text:
                    continue
                eligible += 1
                candidate = {**row, "_sample_text": text, "_dataset_file": path.name}
                if len(reservoir) < count:
                    reservoir.append(candidate)
                else:
                    replacement = rng.randrange(eligible)
                    if replacement < count:
                        reservoir[replacement] = candidate
    if not reservoir:
        raise ValueError("no dataset rows match the requested filters")
    # Cropping after selection preserves uniform document probability.
    return [
        Sample(
            document_id=str(row.get("id", "unknown")),
            language=str(row.get("language", "unknown")),
            domain=str(row.get("domain", row.get("category", "unknown"))),
            source_id=str(row.get("source_id", row.get("_dataset_file", "unknown"))),
            text=_crop(row["_sample_text"], maximum_characters, rng),
        )
        for row in reservoir
    ]


def load_adapter(specification: str) -> AmarkenTokenizer:
    """Load NAME=PATH or infer a display name from PATH."""
    if "=" in specification:
        name, raw_path = specification.split("=", 1)
    else:
        raw_path = specification
        name = Path(raw_path).stem
    path = Path(raw_path)
    if not path.is_file():
        raise ValueError(f"tokenizer artifact does not exist: {path}")
    if path.suffix == ".json":
        return AmarkenTokenizer(path, name=name)
    raise ValueError("tokenizer path must end in .json")


def _print_legend(legend: list[dict]) -> None:
    for token in legend:
        print(
            f"  {token['position']:>3}  id={token['id']:<6} "
            f"surface={token['surface']!r} piece={token['piece']!r}"
        )


def run(args: argparse.Namespace) -> None:
    adapters = [load_adapter(specification) for specification in args.tokenizer]
    samples = reservoir_samples(
        args.dataset,
        args.samples,
        args.seed,
        args.max_characters,
        language=args.language,
        domain=args.domain,
    )
    use_color = not args.no_color and sys.stdout.isatty()
    for sample_index, sample in enumerate(samples, 1):
        print("=" * 88)
        print(
            f"sample={sample_index} id={sample.document_id} language={sample.language} "
            f"domain={sample.domain} source={sample.source_id} chars={len(sample.text)}"
        )
        if probable_mojibake(sample.text):
            print(
                "warning=probable-mojibake source text contains encoding-corruption markers"
            )
        for adapter in adapters:
            rendered, legend = render_tokens(adapter, sample.text, color=use_color)
            words = len(sample.text.split())
            print("-" * 88)
            print(
                f"tokenizer={adapter.name} vocab={adapter.vocab_size()} tokens={len(legend)} "
                f"tokens/word={len(legend) / max(words, 1):.3f}"
            )
            print(rendered)
            if args.legend:
                _print_legend(legend)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        action="append",
        default=None,
        metavar="[NAME=]PATH",
        help="repeat to compare tokenizers on identical samples",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSONL file, or shard directory combining shard-*.jsonl and translations-*.jsonl",
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-characters", type=int, default=512)
    parser.add_argument("--language", help="optional exact JSONL language filter")
    parser.add_argument("--domain", help="optional exact JSONL domain filter")
    parser.add_argument(
        "--legend", action="store_true", help="show position/id/piece mapping"
    )
    parser.add_argument(
        "--no-color", action="store_true", help="use bracket boundaries for logs"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tokenizer is None:
        args.tokenizer = [
            "tiktoken=artifacts/tokenizers/v2/tiktoken-style-tr-bpe-12k.json"
        ]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
