import json
from pathlib import Path

from src.tokenization.sweep import train_byte_bpe
from src.tokenization.visualize import (
    dataset_files,
    probable_mojibake,
    render_tokens,
    reservoir_samples,
)


def test_no_color_render_exposes_every_boundary_and_whitespace(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("def f():\n    return 1\nTürkçe metin\n" * 20), encoding="utf-8")
    adapter = train_byte_bpe([corpus], 400, tmp_path / "tokenizer.json")
    rendered, legend = render_tokens(adapter, "def f():\n    return 1", color=False)
    assert 0 < rendered.count("⟦") <= len(legend)
    assert "↵" in rendered and "·" in rendered
    assert all(
        {"position", "id", "piece", "surface", "offset"} == set(token)
        for token in legend
    )


def test_render_uses_exact_source_for_split_utf8_tokens(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("Türkçe 🙂\n" * 20, encoding="utf-8")
    adapter = train_byte_bpe([corpus], 300, tmp_path / "tokenizer.json")
    text = "Türkçe 🙂"
    rendered, _legend = render_tokens(adapter, text, color=False)
    reconstructed = rendered.replace("⟦", "").replace("⟧", "").replace("·", " ")
    assert reconstructed == text


def test_probable_mojibake_detection():
    assert probable_mojibake("Ä±sÄ±nma gerÃ§ek baÅ\x9fladı")
    assert not probable_mojibake("ısınma gerçek başladı")
    assert not probable_mojibake("Mûsâ'nın ve İbrâhim’in sahifeleri")


def test_reservoir_sampling_is_deterministic_and_filterable(tmp_path: Path):
    dataset = tmp_path / "data.jsonl"
    rows = [
        {
            "id": str(i),
            "language": "tr" if i % 2 else "en",
            "domain": "prose",
            "source_id": "x",
            "text": f"text {i}",
        }
        for i in range(20)
    ]
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    first = reservoir_samples(dataset, 4, 7, 100, language="tr")
    second = reservoir_samples(dataset, 4, 7, 100, language="tr")
    assert first == second and len(first) == 4
    assert all(sample.language == "tr" for sample in first)


def test_reservoir_sampling_combines_synthetic_and_translation_shards(tmp_path: Path):
    shards = tmp_path / "shards"
    shards.mkdir()
    synthetic = {
        "id": "syn-1",
        "language": "en",
        "category": "conversation",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    }
    translation = {
        "id": "translation-1",
        "language": "tr",
        "category": "translation",
        "messages": [
            {"role": "user", "content": "Translate hello"},
            {"role": "assistant", "content": "Merhaba"},
        ],
    }
    (shards / "shard-000000.jsonl").write_text(
        json.dumps(synthetic) + "\n", encoding="utf-8"
    )
    (shards / "translations-000000.jsonl").write_text(
        json.dumps(translation) + "\n", encoding="utf-8"
    )
    (shards / "current.partial.json").write_text("not JSONL", encoding="utf-8")

    assert [path.name for path in dataset_files(shards)] == [
        "shard-000000.jsonl",
        "translations-000000.jsonl",
    ]
    samples = reservoir_samples(shards, 2, 2026, 500)
    assert {sample.document_id for sample in samples} == {"syn-1", "translation-1"}
    assert {sample.domain for sample in samples} == {"conversation", "translation"}
    assert any(
        "<|user|>: Translate hello \n<|end|>\n<|assistant|>: Merhaba \n<|end|>"
        in sample.text
        for sample in samples
    )


def test_translation_filter_and_tokenizer_round_trip_across_shards(tmp_path: Path):
    shards = tmp_path / "shards"
    shards.mkdir()
    rows = [
        {
            "id": f"translation-{index}",
            "language": language,
            "category": "translation",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
        }
        for index, (language, prompt, answer) in enumerate(
            [
                ("tr", "Translate: Good morning", "Günaydın"),
                ("en", "Çevir: Nasılsın?", "How are you?"),
            ]
        )
    ]
    (shards / "translations-000000.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(message["content"] for row in rows for message in row["messages"])
        * 30,
        encoding="utf-8",
    )
    adapter = train_byte_bpe([corpus], 300, tmp_path / "tokenizer.json")

    samples = reservoir_samples(shards, 2, 7, 500, domain="translation")
    assert len(samples) == 2
    for sample in samples:
        ids = adapter.encode(sample.text)
        assert ids
        assert adapter.decode(ids) == sample.text
