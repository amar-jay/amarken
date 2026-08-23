from pathlib import Path

import pytest

from src.tokenization import (
    load_tokenizer,
    tokenizer_artifact_bytes,
    tokenizer_fingerprint,
)


def test_runtime_tokenizer_contract():
    path = Path("artifacts/tokenizers/v3/tiktoken-tr-bpe-12k.json")
    tokenizer = load_tokenizer(path)
    probes = [
        "İstanbul'dan Ankara'ya gidiyorum.",
        "def çözüm(x):\n    return x + 1\n",
        "English, Türkçe, emoji 🙂 and tabs\tremain exact.",
    ]
    assert tokenizer.vocab_size() == 12_000
    assert {
        tokenizer.unk_id(),
        tokenizer.bos_id(),
        tokenizer.eos_id(),
        tokenizer.pad_id(),
    } == {0, 1, 2, 3}
    assert all(tokenizer.decode(tokenizer.encode(text)) == text for text in probes)
    assert tokenizer_artifact_bytes(tokenizer) > 0
    assert len(tokenizer_fingerprint(tokenizer)) == 64


def test_runtime_tokenizer_rejects_unknown_artifact(tmp_path):
    path = tmp_path / "tokenizer.txt"
    path.write_text("not a tokenizer")
    with pytest.raises(ValueError, match="unsupported tokenizer artifact"):
        load_tokenizer(path)
