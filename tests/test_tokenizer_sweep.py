import json
from pathlib import Path

from src.tokenization.sweep import (
    TIKTOKEN_TURKISH_PATTERN,
    _write_slice,
    build_training_corpus,
    train_tiktoken_style_bpe,
)

def test_balanced_slice_never_cuts_utf8_or_records(tmp_path: Path):
    destination = tmp_path / "slice.txt"
    result = _write_slice(
        iter(["Türkçe", "English", "fazla uzun kayıt"]), destination, 17
    )
    assert destination.read_text(encoding="utf-8") == "Türkçe\nEnglish\n"
    assert result["records"] == 2 and result["bytes"] == 17


def test_turkish_pattern_prevents_english_contraction_split(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("Ankara'da İstanbul'dan Türkiye'nin İzmir'e\n" * 30), encoding="utf-8"
    )
    adapter = train_tiktoken_style_bpe(
        [corpus],
        400,
        tmp_path / "tr.json",
        pattern=TIKTOKEN_TURKISH_PATTERN,
        name="tr",
    )
    pieces = [
        piece
        for piece, _offset in adapter.tokenizer.pre_tokenizer.pre_tokenize_str(
            "Ankara'da"
        )
    ]
    assert pieces == ["Ankara", "'da"]
    text = "Ankara'da İstanbul'dan Türkiye'nin İzmir'e"
    assert adapter.decode(adapter.encode(text)) == text


def test_synthetic_shards_build_rendered_en_and_tr_corpora(tmp_path: Path):
    shards = tmp_path / "shards"
    shards.mkdir()
    rows = [
        {"language": "en", "messages": [{"role": "user", "content": "Hello"}]},
        {"language": "tr", "messages": [{"role": "assistant", "content": "Merhaba"}]},
    ] * 5
    (shards / "shard-000000.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    corpus, manifest = build_training_corpus(
        {"synthetic_shards": str(shards), "bytes_per_training_slice": 100},
        tmp_path / "out",
    )
    assert [path.stem for path in corpus] == ["en", "tr"]
    assert "<|user|>: Hello" in (tmp_path / "out" / "corpus" / "en.txt").read_text()
    assert "<|assistant|>: Merhaba" in (tmp_path / "out" / "corpus" / "tr.txt").read_text()
    assert set(manifest) == {"en", "tr"}
