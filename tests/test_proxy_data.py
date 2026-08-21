import json
from pathlib import Path

from src.data.proxy import build


def _write_config(root: Path) -> Path:
    config = {
        "schema_version": 1,
        "seed": "test-seed",
        "validation_basis_points": 5000,
        "near_duplicate_hamming_distance": 0,
        "contamination_ngram_tokens": 4,
        "sources": [
            {"id": "en", "kind": "lines", "path": "raw/en.txt", "language": "en", "domain": "text", "license": "test", "url": None, "group_namespace": "pair"},
            {"id": "tr", "kind": "lines", "path": "raw/tr.txt", "language": "tr", "domain": "text", "license": "test", "url": None, "group_namespace": "pair"},
            {"id": "code", "kind": "glob", "path": "code/*.py", "language": "code", "domain": "python", "license": "test", "url": None, "group_namespace": "code"},
        ],
        "contamination_references": ["eval/*.txt"],
    }
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_proxy_build_is_deterministic_grouped_and_contamination_clean(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "code").mkdir()
    (tmp_path / "eval").mkdir()
    # The third EN record is an exact canonical duplicate despite whitespace.
    (tmp_path / "raw/en.txt").write_text("alpha original sentence\nbeta secret benchmark phrase here\n alpha   original sentence \n", encoding="utf-8")
    (tmp_path / "raw/tr.txt").write_text("alfa özgün cümle\nbeta gizli değerlendirme cümlesi\ngama başka cümle\n", encoding="utf-8")
    (tmp_path / "code/module.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    (tmp_path / "eval/heldout.txt").write_text("beta secret benchmark phrase here", encoding="utf-8")
    config = _write_config(tmp_path)

    first = build(config, tmp_path / "out-a", root=tmp_path)
    second = build(config, tmp_path / "out-b", root=tmp_path)
    assert first["outputs"]["train.jsonl"]["sha256"] == second["outputs"]["train.jsonl"]["sha256"]
    assert first["outputs"]["validation.jsonl"]["sha256"] == second["outputs"]["validation.jsonl"]["sha256"]
    assert first["removed"]["exact_duplicate"] == 1
    assert first["contaminated_documents"] == 1
    assert first["distillation"] is False

    clean = _jsonl(tmp_path / "out-a/train.jsonl") + _jsonl(tmp_path / "out-a/validation.jsonl")
    assert all("secret benchmark" not in record["text"] for record in clean)
    # All surviving members of each aligned pair must share a split.
    split_by_id = {record["id"]: split for split in ("train", "validation") for record in _jsonl(tmp_path / f"out-a/{split}.jsonl")}
    groups = {}
    for record in clean:
        previous = groups.setdefault(record["group_id"], split_by_id[record["id"]])
        assert previous == split_by_id[record["id"]]


def test_proxy_rejects_unknown_source_kind(tmp_path):
    (tmp_path / "raw.txt").write_text("text\n", encoding="utf-8")
    config = {
        "schema_version": 1, "seed": "x", "validation_basis_points": 100,
        "near_duplicate_hamming_distance": 3, "contamination_ngram_tokens": 13,
        "sources": [{"id": "bad", "kind": "distillation", "path": "raw.txt", "language": "en", "domain": "synthetic", "license": "none", "url": None, "group_namespace": "bad"}],
        "contamination_references": [],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    try:
        build(path, tmp_path / "out", root=tmp_path)
    except ValueError as error:
        assert "unsupported source kind" in str(error)
    else:
        raise AssertionError("distillation source kind was accepted")
