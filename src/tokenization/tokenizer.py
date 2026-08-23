"""The single tokenizer implementation used throughout Amarken."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer


SPECIAL_TOKENS = (
    "<unk>",
    "<s>",
    "</s>",
    "<pad>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|code|>",
)

CORE_SPECIAL_TOKENS = {
    "unk": "<unk>",
    "bos": "<s>",
    "eos": "</s>",
    "pad": "<pad>",
}


class AmarkenTokenizer:
    """Validated JSON tokenizer for training, evaluation, and inference."""

    kind = "tokenizers-json"

    def __init__(self, path: str | Path, name: str | None = None):
        self.path = Path(path)
        self.name = name or self.path.stem
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if self.path.suffix.lower() != ".json":
            raise ValueError(
                f"unsupported tokenizer artifact: {self.path}; expected .json"
            )
        try:
            self.tokenizer = Tokenizer.from_file(str(self.path))
        except Exception as error:
            raise ValueError(f"invalid tokenizer artifact: {self.path}") from error
        missing = [
            token for token in SPECIAL_TOKENS
            if self.tokenizer.token_to_id(token) is None
        ]
        if missing:
            raise ValueError(f"tokenizer lacks required special tokens: {missing}")
        if len({self._special_id(name) for name in CORE_SPECIAL_TOKENS}) != len(CORE_SPECIAL_TOKENS):
            raise ValueError("tokenizer special-token IDs must be distinct")

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return list(encoded.ids), list(encoded.offsets)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def token_to_id(self, token: str) -> int | None:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        return self.tokenizer.id_to_token(token_id) or ""

    def piece(self, token_id: int) -> str:
        return self.id_to_token(token_id)

    def _special_id(self, name: str) -> int:
        token_id = self.token_to_id(CORE_SPECIAL_TOKENS[name])
        if token_id is None:
            raise ValueError(f"tokenizer lacks {CORE_SPECIAL_TOKENS[name]}")
        return token_id

    def unk_id(self) -> int: return self._special_id("unk")
    def bos_id(self) -> int: return self._special_id("bos")
    def eos_id(self) -> int: return self._special_id("eos")
    def pad_id(self) -> int: return self._special_id("pad")

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return (self.path,)


def load_tokenizer(path: str | Path, name: str | None = None) -> AmarkenTokenizer:
    return AmarkenTokenizer(path, name=name)


def tokenizer_fingerprint(tokenizer: AmarkenTokenizer) -> str:
    digest = hashlib.sha256()
    for path in tokenizer.artifact_paths:
        digest.update(path.name.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def tokenizer_artifact_bytes(tokenizer: AmarkenTokenizer) -> int:
    return sum(path.stat().st_size for path in tokenizer.artifact_paths)
