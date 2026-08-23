"""Scriptable CLI and interactive terminal UI for Amarken checkpoints."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
import random
import shlex
import sys
import time
from typing import Iterator, Literal

# Deterministic CUDA matrix multiplication requires this before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from src.models import AmarkenCausalLM, create_config, create_model
from src.tokenization import AmarkenTokenizer, load_tokenizer, tokenizer_fingerprint

Precision = Literal["fp32", "bf16", "fp16"]


@dataclass(frozen=True)
class LoadedModel:
    """Normalized inference objects and provenance from any checkpoint envelope."""

    model: AmarkenCausalLM
    tokenizer: AmarkenTokenizer
    checkpoint_kind: str
    checkpoint_step: int
    metadata: dict


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return len(self.token_ids) / self.seconds if self.seconds > 0 else float("inf")


def _checkpoint_fields(payload: dict) -> tuple[str, dict, dict, int, str, dict]:
    """Normalize trainer, model-only tournament, and standalone model payloads."""
    if "model_type" not in payload or "model_state" not in payload:
        raise ValueError("checkpoint lacks model_type or model_state")
    if "model_config" in payload:
        # Exact trainer and scaling artifacts both use model_config; the explicit
        # kind distinguishes deploy-only files from resumable optimizer snapshots.
        config = payload["model_config"]
        kind = payload.get("checkpoint_kind", "trainer")
        state = payload.get("trainer_state", {})
        step = int(state.get("update_step", payload.get("step", 0)))
    elif "config" in payload:
        config = payload["config"]
        kind = "standalone"
        step = int(payload.get("step", 0))
    else:
        raise ValueError("checkpoint lacks config/model_config")
    return (
        payload["model_type"],
        config,
        payload["model_state"],
        step,
        kind,
        dict(payload.get("metadata", {})),
    )


def resolve_device(name: str) -> torch.device:
    """Resolve auto conservatively: CUDA when usable, otherwise CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def resolve_precision(requested: str, device: torch.device) -> Precision:
    if requested == "auto":
        # BF16 has FP32-like exponent range and is supported by the tested CUDA
        # environment; CPU defaults to FP32 to avoid backend-dependent slowdowns.
        return "bf16" if device.type == "cuda" else "fp32"
    if requested == "fp16" and device.type != "cuda":
        raise ValueError("fp16 inference requires CUDA")
    return requested  # type: ignore[return-value]


def load_model(
    checkpoint: str | Path, tokenizer: str | Path, device: torch.device
) -> LoadedModel:
    """Load and strictly validate any repository-produced checkpoint."""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    model_type, config_values, state, step, kind, metadata = _checkpoint_fields(payload)
    config = create_config(model_type, **config_values)
    processor = load_tokenizer(tokenizer)
    if processor.vocab_size() != config.vocab_size:
        raise ValueError(
            f"tokenizer vocabulary ({processor.vocab_size()}) does not match model ({config.vocab_size})"
        )
    expected_fingerprint = metadata.get("tokenizer_fingerprint")
    if (
        expected_fingerprint
        and tokenizer_fingerprint(processor) != expected_fingerprint
    ):
        raise ValueError(
            "checkpoint tokenizer fingerprint does not match the supplied tokenizer"
        )
    model = create_model(config)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return LoadedModel(model, processor, kind, step, metadata)


class InferenceSession:
    """Stateful prompt construction and deterministic autoregressive inference."""

    def __init__(
        self,
        loaded: LoadedModel,
        device: torch.device,
        precision: Precision,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int | None = None,
        seed: int = 2026,
        chat: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        if max_new_tokens < 1 or temperature < 0:
            raise ValueError(
                "max_new_tokens must be positive and temperature nonnegative"
            )
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        self.loaded, self.device, self.precision = loaded, device, precision
        self.max_new_tokens, self.temperature, self.top_k = (
            max_new_tokens,
            temperature,
            top_k,
        )
        self.seed, self.chat, self.system_prompt = seed, chat, system_prompt
        self.turns: list[tuple[str, str]] = []

    def reset(self) -> None:
        self.turns.clear()

    def _prompt(self, user_text: str) -> str:
        if not self.chat:
            return user_text
        parts = []
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}\n")
        for user, assistant in self.turns:
            parts.append(f"User: {user}\nAssistant: {assistant}\n")
        parts.append(f"User: {user_text}\nAssistant:")
        return "".join(parts)

    def _autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def generate(self, user_text: str) -> GenerationResult:
        prompt = self._prompt(user_text)
        ids = self.loaded.tokenizer.encode(prompt)
        if not ids:
            raise ValueError("prompt encodes to zero tokens")
        # Keep the newest context because every model is causal and generation's
        # no-cache fallback applies the identical left crop after each new token.
        ids = ids[-self.loaded.model.config.max_position_embeddings :]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        generator_device = (
            self.device if self.device.type == "cuda" else torch.device("cpu")
        )
        generator = torch.Generator(device=generator_device).manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self._autocast():
            output = self.loaded.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                eos_token_id=self.loaded.tokenizer.eos_id(),
                generator=generator,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        generated = output[0, input_ids.shape[1] :].tolist()
        # Retaining token IDs makes whitespace/debug behavior auditable.
        text = self.loaded.tokenizer.decode(generated)
        if self.chat:
            self.turns.append((user_text, text))
        return GenerationResult(text, tuple(generated), len(ids), elapsed)


def _iter_checkpoint_files(root: Path) -> Iterator[Path]:
    for pattern in ("*.pt", "*.pth", "*.ckpt"):
        yield from root.rglob(pattern)


def _print_model_info(
    loaded: LoadedModel, device: torch.device, precision: Precision
) -> None:
    stats = loaded.model.stats(min(512, loaded.model.config.max_position_embeddings))
    variant = loaded.metadata.get("variant", loaded.model.config.model_type)
    print(
        f"model={loaded.model.config.model_type} variant={variant} checkpoint={loaded.checkpoint_kind} "
        f"step={loaded.checkpoint_step} params={stats.total_parameters:,} context={loaded.model.config.max_position_embeddings} "
        f"device={device} precision={precision}"
    )


def _help() -> None:
    print(
        "Commands:\n"
        " /help show commands\n"
        " /reset clear conversation history\n"
        " /settings show generation settings\n"
        " /max-new N set generated-token limit\n"
        " /temperature X set sampling temperature; 0 is greedy\n"
        " /top-k N|none restrict sampling candidates\n"
        " /seed N set deterministic sampling seed\n"
        " /quit exit\n"
    )


def _command(session: InferenceSession, line: str) -> bool:
    pieces = shlex.split(line)
    command = pieces[0].lower()
    if command in ("/quit", "/exit"):
        return False
    if command == "/help":
        _help()
    elif command == "/reset":
        session.reset()
        print("history cleared")
    elif command == "/settings":
        print(
            f"max_new={session.max_new_tokens} temperature={session.temperature} "
            f"top_k={session.top_k} seed={session.seed} chat={session.chat}"
        )
    elif command == "/max-new" and len(pieces) == 2:
        session.max_new_tokens = max(1, int(pieces[1]))
    elif command == "/temperature" and len(pieces) == 2:
        session.temperature = max(0.0, float(pieces[1]))
    elif command == "/top-k" and len(pieces) == 2:
        session.top_k = None if pieces[1].lower() == "none" else max(1, int(pieces[1]))
    elif command == "/seed" and len(pieces) == 2:
        session.seed = int(pieces[1])
    else:
        print("unknown or malformed command; use /help")
    return True


def interactive(session: InferenceSession) -> None:
    """Readline-enabled REPL when available; gracefully supports plain terminals."""
    try:
        import readline  # noqa: F401 - importing activates arrow-key line editing/history.
    except ImportError:
        pass
    print("Amarken inference — /help for commands, /quit to exit")
    if session.chat:
        print(
            "warning: chat formatting is experimental; this checkpoint is not instruction-tuned"
        )
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            try:
                if not _command(session, line):
                    break
            except ValueError as error:
                print(f"error: {error}")
            continue
        try:
            result = session.generate(line)
            print(f"model> {result.text}")
            print(
                f"[{len(result.token_ids)} tokens, {result.seconds:.3f}s, {result.tokens_per_second:.1f} tok/s]"
            )
        except (RuntimeError, ValueError) as error:
            print(f"error: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="trainer, standalone, or model-only .pt checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("artifacts/tokenizers/v2/tiktoken-style-tr-bpe-12k.json"),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto"
    )
    parser.add_argument(
        "--prompt",
        help="one-shot prompt; omitted enters interactive mode when stdin is a TTY",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="retain turns using plain User/Assistant labels",
    )
    parser.add_argument("--system", help="optional system text used only with --chat")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--show-info", action="store_true")
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="write timing/token statistics to stderr",
    )
    parser.add_argument(
        "--list-checkpoints",
        type=Path,
        metavar="DIR",
        help="list checkpoint files and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_checkpoints:
        for path in sorted(_iter_checkpoint_files(args.list_checkpoints)):
            print(path)
        return 0
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --list-checkpoints is used")
    if args.system and not args.chat:
        raise SystemExit("--system requires --chat")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    loaded = load_model(args.checkpoint, args.tokenizer, device)
    session = InferenceSession(
        loaded,
        device,
        precision,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        chat=args.chat,
        system_prompt=args.system,
    )
    if args.show_info:
        _print_model_info(loaded, device, precision)
    prompt = args.prompt
    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read()
    if prompt is None:
        interactive(session)
        return 0
    result = session.generate(prompt)
    print(result.text)
    if args.show_stats:
        print(
            f"prompt_tokens={result.prompt_tokens} generated_tokens={len(result.token_ids)} "
            f"seconds={result.seconds:.6f} tokens_per_second={result.tokens_per_second:.3f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
