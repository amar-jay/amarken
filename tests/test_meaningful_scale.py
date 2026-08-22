import json
from collections import Counter
from pathlib import Path

from src.evaluation.build_benchmark_v2 import build
from src.evaluation.meaningful_scale import _normalize_exact, _wilson


def test_benchmark_v2_is_large_balanced_and_reproducible(tmp_path):
    validation = tmp_path / "validation.jsonl"
    rows = []
    for language in ("en", "tr", "code"):
        for index in range(35):
            rows.append({"id": f"{language}-{index}", "language": language, "text": (f"{language} deterministic heldout text {index}. " * 8)})
    validation.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    a, b = build(validation, first), build(validation, second)
    assert a == b
    assert first.read_bytes() == second.read_bytes()
    assert len(a["multiple_choice"]) == 240
    assert len(a["generative"]) == 120
    assert len(a["language_model"]) == 90
    assert Counter(task["options"].index(task["answer"]) for task in a["multiple_choice"]) == {0: 60, 1: 60, 2: 60, 3: 60}


def test_confidence_intervals_and_exact_normalization():
    interval = _wilson(50, 100)
    assert interval["low"] < 0.5 < interval["high"]
    assert _normalize_exact("  İstanbul\n  açık  ") == "İstanbul açık"
