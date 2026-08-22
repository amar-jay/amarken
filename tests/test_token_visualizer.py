import json
from pathlib import Path

from src.tokenization.v2_sweep import train_byte_bpe
from src.tokenization.visualize import probable_mojibake, render_tokens, reservoir_samples


def test_no_color_render_exposes_every_boundary_and_whitespace(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("def f():\n    return 1\nTürkçe metin\n" * 20), encoding="utf-8")
    adapter = train_byte_bpe([corpus], 400, tmp_path / "tokenizer.json")
    rendered, legend = render_tokens(adapter, "def f():\n    return 1", color=False)
    assert 0 < rendered.count("⟦") <= len(legend)
    assert "↵" in rendered and "·" in rendered
    assert all({"position", "id", "piece", "surface", "offset"} == set(token) for token in legend)


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
        {"id": str(i), "language": "tr" if i % 2 else "en", "domain": "prose", "source_id": "x", "text": f"text {i}"}
        for i in range(20)
    ]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first = reservoir_samples(dataset, 4, 7, 100, language="tr")
    second = reservoir_samples(dataset, 4, 7, 100, language="tr")
    assert first == second and len(first) == 4
    assert all(sample.language == "tr" for sample in first)
