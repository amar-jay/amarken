"""Textual workbench for Amarken checkpoint inference.

Wraps the same session, checkpoint, and sampling contract as ``infer.py``,
but replaces the readline REPL with a two-pane terminal UI.

Usage (from the repository root, next to ``infer.py``)::

    pip install textual
    python amarken_tui.py --checkpoint path/to/model.pt
    python amarken_tui.py --demo
    python amarken_tui.py --scan artifacts/checkpoints

Slash commands from the original REPL still work in the composer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import random
import shlex
import time
from typing import Any, Iterator, Literal

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
    Switch,
    TextArea,
)

Precision = Literal["fp32", "bf16", "fp16"]


# ---------------------------------------------------------------------------
# Backend adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    seconds: float

    @property
    def tokens_per_second(self) -> float:
        return len(self.token_ids) / self.seconds if self.seconds > 0 else float("inf")


@dataclass(frozen=True)
class ModelCard:
    model_type: str
    variant: str
    checkpoint_kind: str
    checkpoint_step: int
    parameters: int | None
    context: int | None
    device: str
    precision: str
    checkpoint_path: str
    demo: bool = False


class SessionProtocol:
    """Structural surface the TUI talks to — real or demo."""

    max_new_tokens: int
    temperature: float
    top_k: int | None
    seed: int
    chat: bool
    system_prompt: str | None
    turns: list[tuple[str, str]]

    def reset(self) -> None: ...
    def generate(self, user_text: str) -> GenerationResult: ...


def _iter_checkpoint_files(root: Path) -> Iterator[Path]:
    for pattern in ("*.pt", "*.pth", "*.ckpt"):
        yield from root.rglob(pattern)


def _try_backend() -> Any | None:
    for name in ("infer", "amarken_infer"):
        try:
            return import_module(name)
        except Exception:
            continue
    return None


class DemoSession:
    """Deterministic stand-in so the layout can be exercised without torch."""

    def __init__(
        self,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int | None = None,
        seed: int = 2026,
        chat: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        self.max_new_tokens = max(1, max_new_tokens)
        self.temperature = max(0.0, temperature)
        self.top_k = top_k
        self.seed = seed
        self.chat = chat
        self.system_prompt = system_prompt
        self.turns: list[tuple[str, str]] = []

    def reset(self) -> None:
        self.turns.clear()

    def generate(self, user_text: str) -> GenerationResult:
        started = time.perf_counter()
        # Cheap, reproducible “thinking” delay so the busy state is visible.
        time.sleep(min(0.85, 0.12 + min(len(user_text), 80) * 0.004))
        n = max(8, min(self.max_new_tokens, 24))
        mode = "greedy" if self.temperature == 0 else f"T={self.temperature:g}"
        top = "all" if self.top_k is None else str(self.top_k)
        lines = [
            f"[demo · {mode} · top-k={top} · seed={self.seed}]",
            user_text.strip() or "(empty)",
        ]
        if self.chat:
            lines.append(f"turn={len(self.turns) + 1}  system={'yes' if self.system_prompt else 'no'}")
        text = "\n".join(lines)
        elapsed = max(time.perf_counter() - started, 1e-6)
        ids = tuple((self.seed + i) % 12_000 for i in range(n))
        if self.chat:
            self.turns.append((user_text, text))
        return GenerationResult(text, ids, max(1, len(user_text.encode()) // 4), elapsed)


def load_real_session(args: argparse.Namespace) -> tuple[SessionProtocol, ModelCard]:
    backend = _try_backend()
    if backend is None:
        raise RuntimeError(
            "could not import infer.py — keep this file next to the original CLI "
            "or pass --demo to preview the workbench"
        )
    random.seed(args.seed)
    if hasattr(backend, "torch"):
        backend.torch.manual_seed(args.seed)
    device = backend.resolve_device(args.device)
    precision = backend.resolve_precision(args.precision, device)
    loaded = backend.load_model(args.checkpoint, args.tokenizer, device)
    session = backend.InferenceSession(
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
    parameters = None
    context = None
    model_type = "unknown"
    variant = "unknown"
    try:
        model_type = loaded.model.config.model_type
        context = int(loaded.model.config.max_position_embeddings)
        variant = loaded.metadata.get("variant", model_type)
        stats = loaded.model.stats(min(512, context))
        parameters = int(stats.total_parameters)
    except Exception:
        variant = loaded.metadata.get("variant", model_type)
    card = ModelCard(
        model_type=str(model_type),
        variant=str(variant),
        checkpoint_kind=str(loaded.checkpoint_kind),
        checkpoint_step=int(loaded.checkpoint_step),
        parameters=parameters,
        context=context,
        device=str(device),
        precision=str(precision),
        checkpoint_path=str(args.checkpoint),
        demo=False,
    )
    return session, card


def load_demo_session(args: argparse.Namespace) -> tuple[SessionProtocol, ModelCard]:
    session = DemoSession(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        chat=args.chat,
        system_prompt=args.system,
    )
    card = ModelCard(
        model_type="amarken",
        variant="demo",
        checkpoint_kind="demo",
        checkpoint_step=0,
        parameters=12_582_912,
        context=2048,
        device="cpu",
        precision="fp32",
        checkpoint_path="(demo — no weights loaded)",
        demo=True,
    )
    return session, card


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


HELP_BODY = """\
[b]Composer[/b]
  Enter            send prompt
  /help            this overlay
  /reset           clear conversation
  /settings        reprint live knobs
  /max-new N       generated-token cap
  /temperature X   0 is greedy
  /top-k N|none    sampling shortlist
  /seed N          deterministic RNG
  /quit            leave the workbench

[b]Keys[/b]
  ctrl+q           quit
  ctrl+r           reset history
  ctrl+l           focus composer
  f1               help
  f2               focus settings
  f3               scan checkpoints

[b]Notes[/b]
  Chat mode prefixes turns with User/Assistant labels.
  The original checkpoints are not instruction-tuned; chat is experimental.
  Generation runs on a worker thread so the UI stays responsive.
"""


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss", "close"),
        Binding("f1", "dismiss", "close", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Label("Amarken workbench", id="help-title")
            yield Static(HELP_BODY, id="help-body", markup=True)
            yield Button("Close", id="help-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-close":
            self.dismiss()


class ScanScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "dismiss", "close")]

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root

    def compose(self) -> ComposeResult:
        found = sorted(_iter_checkpoint_files(self.root)) if self.root.exists() else []
        with Vertical(id="scan-dialog"):
            yield Label(f"Checkpoints under {self.root}", id="scan-title")
            log = RichLog(id="scan-log", wrap=True, highlight=False, markup=True)
            yield log
            yield Button("Close", id="scan-close", variant="primary")
            if not found:
                log.write(Text.from_markup("[dim]no .pt / .pth / .ckpt files found[/]"))
            else:
                for path in found:
                    log.write(Text(str(path)))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "scan-close":
            self.dismiss()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class GenerationDone(Message):
    def __init__(self, prompt: str, result: GenerationResult) -> None:
        super().__init__()
        self.prompt = prompt
        self.result = result


class GenerationFailed(Message):
    def __init__(self, prompt: str, error: str) -> None:
        super().__init__()
        self.prompt = prompt
        self.error = error


class AmarkenTUI(App[None]):
    TITLE = "Amarken"
    SUB_TITLE = "checkpoint workbench"
    CSS = """
    Screen {
        background: #111;
        color: #ddd;
    }

    Header, Footer {
        background: #111;
        color: #999;
    }

    #banner {
        dock: top;
        height: auto;
        padding: 0 1;
        color: #aaa;
        border-bottom: solid #333;
    }

    #body {
        height: 1fr;
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 32;
        grid-gutter: 0;
    }

    #transcript-wrap {
        height: 1fr;
        border-right: solid #333;
    }

    #transcript-label {
        padding: 0 1;
        color: #888;
    }

    #transcript {
        height: 1fr;
        padding: 0 1 1 1;
        background: #111;
        scrollbar-background: #111;
        scrollbar-color: #444;
    }

    #sidebar {
        height: 1fr;
        padding: 0 1 1 1;
        overflow-y: auto;
    }

    #sidebar-title, .field-label {
        color: #888;
        padding: 1 0 0 0;
    }

    #sidebar-title {
        padding-top: 0;
    }

    Input, TextArea {
        background: #111;
        color: #ddd;
        border: solid #333;
    }

    Input:focus, TextArea:focus {
        border: solid #666;
    }

    #system {
        height: 7;
    }

    .row {
        height: auto;
        align: left middle;
    }

    #chat-label {
        width: auto;
        color: #888;
        padding-right: 1;
    }

    Button {
        background: #222;
        color: #ddd;
        border: solid #333;
        min-width: 10;
    }

    Button:hover {
        background: #2a2a2a;
    }

    Button.-primary {
        background: #2a2a2a;
        color: #eee;
    }

    #actions {
        height: auto;
        padding-top: 1;
    }

    #composer-wrap {
        dock: bottom;
        height: auto;
        border-top: solid #333;
        padding: 1;
    }

    #composer-row {
        height: auto;
    }

    #prompt {
        width: 1fr;
    }

    #send {
        width: 12;
        margin-left: 1;
    }

    #status {
        height: 1;
        color: #888;
        padding: 0 1 0 0;
    }

    #status.busy {
        color: #bbb;
    }

    #status.error {
        color: #c66;
    }

    HelpScreen, ScanScreen {
        align: center middle;
        background: #000 60%;
    }

    #help-dialog, #scan-dialog {
        width: 72;
        max-width: 100%;
        height: auto;
        max-height: 90%;
        background: #111;
        border: solid #444;
        padding: 1 2;
    }

    #help-title, #scan-title {
        color: #ccc;
        padding-bottom: 1;
    }

    #help-body, #scan-log {
        height: auto;
        max-height: 24;
        background: #111;
    }

    #help-close, #scan-close {
        margin-top: 1;
        width: 12;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "quit"),
        Binding("ctrl+c", "quit", "quit", show=False),
        Binding("ctrl+r", "reset", "reset"),
        Binding("ctrl+l", "focus_prompt", "compose"),
        Binding("f1", "help", "help"),
        Binding("f2", "focus_settings", "settings"),
        Binding("f3", "scan", "scan"),
    ]

    busy: reactive[bool] = reactive(False)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.session: SessionProtocol | None = None
        self.card: ModelCard | None = None
        self._load_error: str | None = None
        self._last_stats: str = "idle"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("loading checkpoint…", id="banner")
        with Horizontal(id="body"):
            with Vertical(id="transcript-wrap"):
                yield Label("conversation", id="transcript-label")
                yield RichLog(id="transcript", wrap=True, highlight=False, markup=True)
            with VerticalScroll(id="sidebar"):
                yield Label("sampling", id="sidebar-title")
                yield Label("max new tokens", classes="field-label")
                yield Input(value=str(self.args.max_new_tokens), id="max-new", type="integer")
                yield Label("temperature  (0 = greedy)", classes="field-label")
                yield Input(value=str(self.args.temperature), id="temperature")
                yield Label("top-k  (none disables)", classes="field-label")
                yield Input(
                    value="none" if self.args.top_k is None else str(self.args.top_k),
                    id="top-k",
                )
                yield Label("seed", classes="field-label")
                yield Input(value=str(self.args.seed), id="seed", type="integer")
                yield Label("chat mode", classes="field-label")
                with Horizontal(classes="row"):
                    yield Switch(value=bool(self.args.chat), id="chat")
                    yield Label("retain turns", id="chat-label")
                yield Label("system prompt  (chat only)", classes="field-label")
                yield TextArea(self.args.system or "", id="system")
                with Horizontal(id="actions"):
                    yield Button("Apply", id="apply", variant="primary")
                    yield Button("Reset", id="reset")
                    yield Button("Help", id="help")
        with Vertical(id="composer-wrap"):
            yield Static("ready", id="status")
            with Horizontal(id="composer-row"):
                yield Input(placeholder="prompt or /command  —  enter to send", id="prompt")
                yield Button("Send", id="send", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()
        self._boot()

    @work(thread=True, exclusive=True)
    def _boot(self) -> None:
        try:
            if self.args.demo:
                session, card = load_demo_session(self.args)
            else:
                session, card = load_real_session(self.args)
        except Exception as error:
            self._load_error = str(error)
            self.call_from_thread(self._on_boot_failed, str(error))
            return
        self.session = session
        self.card = card
        self.call_from_thread(self._on_boot_ok)

    def _on_boot_ok(self) -> None:
        assert self.card is not None and self.session is not None
        banner = self.query_one("#banner", Static)
        banner.update(self._banner_text(self.card))
        banner.set_class(self.card.demo, "demo")
        log = self.query_one("#transcript", RichLog)
        log.write(Text.from_markup("[b]Amarken inference[/]  —  /help · F1"))
        if self.card.demo:
            log.write(Text.from_markup("[dim]demo session — replies are local stubs, not model output[/]"))
        if self.session.chat:
            log.write(
                Text.from_markup(
                    "warning: chat formatting is experimental; "
                    "this checkpoint is not instruction-tuned"
                )
            )
        self._set_status("ready")
        self.sub_title = "demo" if self.card.demo else self.card.checkpoint_kind

    def _on_boot_failed(self, error: str) -> None:
        banner = self.query_one("#banner", Static)
        banner.update(f"failed to load  ·  {error}")
        self.query_one("#transcript", RichLog).write(Text.from_markup(f"[red]error:[/] {error}"))
        self._set_status(f"error: {error}", kind="error")
        self.notify(error, severity="error", timeout=8)

    def _banner_text(self, card: ModelCard) -> str:
        params = f"{card.parameters:,}" if card.parameters is not None else "?"
        context = str(card.context) if card.context is not None else "?"
        return (
            f"model={card.model_type}  variant={card.variant}  "
            f"checkpoint={card.checkpoint_kind}  step={card.checkpoint_step}  "
            f"params={params}  context={context}  "
            f"device={card.device}  precision={card.precision}"
        )

    def _set_status(self, text: str, *, kind: str = "") -> None:
        status = self.query_one("#status", Static)
        status.update(text)
        status.remove_class("busy")
        status.remove_class("error")
        if kind:
            status.add_class(kind)

    def _read_settings(self) -> dict[str, Any]:
        max_new = max(1, int(self.query_one("#max-new", Input).value or "1"))
        temperature = max(0.0, float(self.query_one("#temperature", Input).value or "0"))
        raw_k = (self.query_one("#top-k", Input).value or "none").strip().lower()
        top_k: int | None
        if raw_k in {"", "none", "off", "-"}:
            top_k = None
        else:
            top_k = max(1, int(raw_k))
        seed = int(self.query_one("#seed", Input).value or "0")
        chat = bool(self.query_one("#chat", Switch).value)
        system = self.query_one("#system", TextArea).text.strip() or None
        return {
            "max_new_tokens": max_new,
            "temperature": temperature,
            "top_k": top_k,
            "seed": seed,
            "chat": chat,
            "system_prompt": system,
        }

    def _apply_settings(self, *, silent: bool = False) -> bool:
        if self.session is None:
            self.notify("model is still loading", severity="warning")
            return False
        try:
            values = self._read_settings()
        except ValueError as error:
            self.notify(f"invalid setting: {error}", severity="error")
            return False
        self.session.max_new_tokens = values["max_new_tokens"]
        self.session.temperature = values["temperature"]
        self.session.top_k = values["top_k"]
        self.session.seed = values["seed"]
        self.session.chat = values["chat"]
        self.session.system_prompt = values["system_prompt"]
        if values["system_prompt"] and not values["chat"]:
            self.notify("--system is only used when chat mode is on", severity="warning")
        if not silent:
            self._set_status(self._settings_line())
            self.notify("settings applied")
        return True

    def _sync_widgets_from_session(self) -> None:
        if self.session is None:
            return
        self.query_one("#max-new", Input).value = str(self.session.max_new_tokens)
        self.query_one("#temperature", Input).value = str(self.session.temperature)
        self.query_one("#top-k", Input).value = "none" if self.session.top_k is None else str(self.session.top_k)
        self.query_one("#seed", Input).value = str(self.session.seed)
        self.query_one("#chat", Switch).value = bool(self.session.chat)
        self.query_one("#system", TextArea).text = self.session.system_prompt or ""

    def _settings_line(self) -> str:
        if self.session is None:
            return "no session"
        return (
            f"max_new={self.session.max_new_tokens}  temperature={self.session.temperature}  "
            f"top_k={self.session.top_k}  seed={self.session.seed}  chat={self.session.chat}"
        )

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_reset(self) -> None:
        if self.session is None:
            return
        self.session.reset()
        log = self.query_one("#transcript", RichLog)
        log.clear()
        log.write(Text.from_markup("[dim]history cleared[/]"))
        self._set_status("history cleared")

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt", Input).focus()

    def action_focus_settings(self) -> None:
        self.query_one("#max-new", Input).focus()

    def action_scan(self) -> None:
        root = self.args.scan or Path(".")
        self.push_screen(ScanScreen(Path(root)))

    @on(Button.Pressed, "#apply")
    def _on_apply(self) -> None:
        self._apply_settings()

    @on(Button.Pressed, "#reset")
    def _on_reset_button(self) -> None:
        self.action_reset()

    @on(Button.Pressed, "#help")
    def _on_help_button(self) -> None:
        self.action_help()

    @on(Button.Pressed, "#send")
    def _on_send_button(self) -> None:
        self._submit()

    @on(Input.Submitted, "#prompt")
    def _on_prompt_submitted(self) -> None:
        self._submit()

    def _submit(self) -> None:
        box = self.query_one("#prompt", Input)
        line = box.value.strip()
        if not line:
            return
        if self.busy:
            self.notify("generation already running", severity="warning")
            return
        if line.startswith("/"):
            box.value = ""
            self._handle_command(line)
            return
        if self.session is None:
            self.notify(self._load_error or "model is still loading", severity="error")
            return
        if not self._apply_settings(silent=True):
            return
        box.value = ""
        self._append_user(line)
        self.busy = True
        self._set_status("generating…", kind="busy")
        self.query_one("#send", Button).disabled = True
        self._generate(line)

    def _handle_command(self, line: str) -> bool:
        if self.session is None and not line.lower().split()[0] in {"/help", "/quit", "/exit"}:
            self.notify("model is still loading", severity="warning")
            return True
        try:
            pieces = shlex.split(line)
        except ValueError as error:
            self.notify(f"malformed command: {error}", severity="error")
            return True
        command = pieces[0].lower()
        try:
            if command in ("/quit", "/exit"):
                self.exit()
                return False
            if command == "/help":
                self.action_help()
            elif command == "/reset":
                self.action_reset()
            elif command == "/settings":
                self._set_status(self._settings_line())
                self.query_one("#transcript", RichLog).write(Text.from_markup(f"[dim]{self._settings_line()}[/]"))
            elif command == "/max-new" and len(pieces) == 2 and self.session is not None:
                self.session.max_new_tokens = max(1, int(pieces[1]))
                self._sync_widgets_from_session()
                self._set_status(self._settings_line())
            elif command == "/temperature" and len(pieces) == 2 and self.session is not None:
                self.session.temperature = max(0.0, float(pieces[1]))
                self._sync_widgets_from_session()
                self._set_status(self._settings_line())
            elif command == "/top-k" and len(pieces) == 2 and self.session is not None:
                self.session.top_k = None if pieces[1].lower() == "none" else max(1, int(pieces[1]))
                self._sync_widgets_from_session()
                self._set_status(self._settings_line())
            elif command == "/seed" and len(pieces) == 2 and self.session is not None:
                self.session.seed = int(pieces[1])
                self._sync_widgets_from_session()
                self._set_status(self._settings_line())
            else:
                self.notify("unknown or malformed command — F1 for help", severity="warning")
        except ValueError as error:
            self.notify(str(error), severity="error")
        return True

    def _append_user(self, text: str) -> None:
        log = self.query_one("#transcript", RichLog)
        log.write(Text())
        log.write(Text.from_markup("[b]you[/]"))
        log.write(Text(text))

    def _append_model(self, result: GenerationResult) -> None:
        log = self.query_one("#transcript", RichLog)
        log.write(Text.from_markup("[b]model[/]"))
        log.write(Text(result.text or "[empty]"))
        stats = (
            f"{len(result.token_ids)} tokens  ·  {result.seconds:.3f}s  ·  "
            f"{result.tokens_per_second:.1f} tok/s  ·  prompt={result.prompt_tokens}"
        )
        log.write(Text(stats, style="dim"))
        self._last_stats = stats

    @work(thread=True, exclusive=True)
    def _generate(self, prompt: str) -> None:
        assert self.session is not None
        try:
            result = self.session.generate(prompt)
        except Exception as error:
            self.post_message(GenerationFailed(prompt, str(error)))
            return
        # Normalize whatever the backend returned into our dataclass.
        if not isinstance(result, GenerationResult):
            result = GenerationResult(
                text=result.text,
                token_ids=tuple(result.token_ids),
                prompt_tokens=int(result.prompt_tokens),
                seconds=float(result.seconds),
            )
        self.post_message(GenerationDone(prompt, result))

    def on_generation_done(self, event: GenerationDone) -> None:
        self.busy = False
        self.query_one("#send", Button).disabled = False
        self._append_model(event.result)
        self._set_status(self._last_stats)
        self.query_one("#prompt", Input).focus()

    def on_generation_failed(self, event: GenerationFailed) -> None:
        self.busy = False
        self.query_one("#send", Button).disabled = False
        self.query_one("#transcript", RichLog).write(Text.from_markup(f"[red]error:[/] {event.error}"))
        self._set_status(f"error: {event.error}", kind="error")
        self.notify(event.error, severity="error")
        self.query_one("#prompt", Input).focus()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="trainer, standalone, or model-only .pt checkpoint")
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizers/amarken-en-tr-12k.model"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--chat", action="store_true", help="retain turns using plain User/Assistant labels")
    parser.add_argument("--system", help="optional system text used only with --chat")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--demo", action="store_true", help="run the workbench without loading weights")
    parser.add_argument(
        "--scan",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="directory F3 searches for .pt / .pth / .ckpt files",
    )
    parser.add_argument("--list-checkpoints", type=Path, metavar="DIR", help="list checkpoint files and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_checkpoints:
        for path in sorted(_iter_checkpoint_files(args.list_checkpoints)):
            print(path)
        return 0
    if args.system and not args.chat:
        raise SystemExit("--system requires --chat")
    if not args.demo and args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --demo or --list-checkpoints is used")
    AmarkenTUI(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
