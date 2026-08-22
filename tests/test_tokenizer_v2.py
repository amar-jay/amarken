from pathlib import Path

from src.tokenization.v2_sweep import _write_slice, train_byte_bpe


def test_balanced_slice_never_cuts_utf8_or_records(tmp_path: Path):
    destination = tmp_path / "slice.txt"
    result = _write_slice(iter(["Türkçe", "English", "fazla uzun kayıt"]), destination, 17)
    assert destination.read_text(encoding="utf-8") == "Türkçe\nEnglish\n"
    assert result["records"] == 2 and result["bytes"] == 17


def test_byte_bpe_roundtrips_code_and_turkish(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("def f(x):\n    return x + 1\nTürkiye'nin başkenti Ankara'dır.\n" * 20),
        encoding="utf-8",
    )
    adapter = train_byte_bpe([corpus], 400, tmp_path / "tokenizer.json")
    for text in ("def f(x):\n    return x + 1\n", "evlerimizdekilerden", "🙂\n"):
        assert adapter.decode(adapter.encode(text)) == text
    indented = len(adapter.encode("\n    return value\n"))
    unindented = len(adapter.encode("\nreturn value\n"))
    assert indented - unindented <= 1
