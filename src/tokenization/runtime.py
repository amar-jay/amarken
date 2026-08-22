"""Runtime tokenizer contract shared by training, evaluation, and inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import sentencepiece as spm
from tokenizers import Tokenizer


SPECIAL_TOKEN_NAMES = {"unk": "<unk>", "bos": "<s>", "eos": "</s>", "pad": "<pad>"}


@runtime_checkable
class RuntimeTokenizer(Protocol):
    path: Path
    kind: str
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    def vocab_size(self) -> int: ...
    def token_to_id(self, token: str) -> int | None: ...
    def id_to_token(self, token_id: int) -> str: ...
    def unk_id(self) -> int: ...
    def bos_id(self) -> int: ...
    def eos_id(self) -> int: ...
    def pad_id(self) -> int: ...
    def artifact_paths(self) -> tuple[Path, ...]: ...


class SentencePieceRuntimeTokenizer:
    kind = "sentencepiece"

    def __init__(self, path: Path):
        self.path = path
        self.processor = spm.SentencePieceProcessor(model_file=str(path))

    def encode(self, text: str) -> list[int]: return list(self.processor.encode(text, out_type=int))
    def decode(self, ids: Sequence[int]) -> str: return self.processor.decode(list(ids))
    def vocab_size(self) -> int: return self.processor.vocab_size()
    def id_to_token(self, token_id: int) -> str: return self.processor.id_to_piece(token_id)
    def unk_id(self) -> int: return self.processor.unk_id()
    def bos_id(self) -> int: return self.processor.bos_id()
    def eos_id(self) -> int: return self.processor.eos_id()
    def pad_id(self) -> int: return self.processor.pad_id()

    def token_to_id(self, token: str) -> int | None:
        token_id = int(self.processor.piece_to_id(token))
        return None if token_id == self.unk_id() and token != SPECIAL_TOKEN_NAMES["unk"] else token_id

    def artifact_paths(self) -> tuple[Path, ...]:
        vocab = self.path.with_suffix(".vocab")
        return (self.path, vocab) if vocab.is_file() else (self.path,)


class HuggingFaceRuntimeTokenizer:
    kind = "huggingface-tokenizers"

    def __init__(self, path: Path):
        self.path = path
        self.tokenizer = Tokenizer.from_file(str(path))
        missing = [token for token in SPECIAL_TOKEN_NAMES.values() if self.tokenizer.token_to_id(token) is None]
        if missing:
            raise ValueError(f"tokenizer lacks required special tokens: {missing}")

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def vocab_size(self) -> int: return self.tokenizer.get_vocab_size(with_added_tokens=True)
    def token_to_id(self, token: str) -> int | None: return self.tokenizer.token_to_id(token)
    def id_to_token(self, token_id: int) -> str: return self.tokenizer.id_to_token(token_id) or ""

    def _special_id(self, name: str) -> int:
        token_id = self.token_to_id(SPECIAL_TOKEN_NAMES[name])
        if token_id is None:
            raise ValueError(f"tokenizer lacks {SPECIAL_TOKEN_NAMES[name]}")
        return token_id

    def unk_id(self) -> int: return self._special_id("unk")
    def bos_id(self) -> int: return self._special_id("bos")
    def eos_id(self) -> int: return self._special_id("eos")
    def pad_id(self) -> int: return self._special_id("pad")
    def artifact_paths(self) -> tuple[Path, ...]: return (self.path,)


def load_tokenizer(path: str | Path) -> RuntimeTokenizer:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if resolved.suffix == ".model":
        tokenizer: RuntimeTokenizer = SentencePieceRuntimeTokenizer(resolved)
    elif resolved.suffix == ".json":
        tokenizer = HuggingFaceRuntimeTokenizer(resolved)
    else:
        raise ValueError(f"unsupported tokenizer artifact: {resolved}; expected .model or .json")
    if min(tokenizer.unk_id(), tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.pad_id()) < 0:
        raise ValueError("tokenizer must define nonnegative unk/bos/eos/pad IDs")
    return tokenizer


def tokenizer_fingerprint(tokenizer: RuntimeTokenizer) -> str:
    digest = hashlib.sha256()
    for path in tokenizer.artifact_paths():
        digest.update(path.name.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def tokenizer_artifact_bytes(tokenizer: RuntimeTokenizer) -> int:
    return sum(path.stat().st_size for path in tokenizer.artifact_paths())
