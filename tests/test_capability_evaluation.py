import json
from collections import Counter
from pathlib import Path

from src.evaluation.capability import _contamination_scan


def test_frozen_benchmark_is_balanced_and_mechanically_scored():
    benchmark = json.loads(Path("benchmarks/proxy_capability_v1.json").read_text())
    tasks = benchmark["tasks"]
    assert len(tasks) == 30 and len({task["id"] for task in tasks}) == 30
    assert {task["target"] for task in tasks} <= {"A", "B", "C", "D"}
    # Balanced answer positions prevent a constant-letter policy from appearing
    # capable; 30 items permit only an 8/8/7/7 near-balance.
    target_counts = Counter(task["target"] for task in tasks)
    assert max(target_counts.values()) - min(target_counts.values()) <= 1
    categories = {task["category"] for task in tasks}
    assert categories == {"instruction_following", "compositional_reasoning", "retrieval", "state_tracking", "tool_syntax"}
    for category in categories:
        for language in ("en", "tr"):
            assert sum(task["category"] == category and task["language"] == language for task in tasks) == 3


def test_contamination_scan_detects_long_exact_windows(tmp_path: Path):
    phrase = "one two three four five six seven eight nine ten eleven twelve thirteen"
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({"tasks": [{"prompt": phrase}]}), encoding="utf-8")
    train = tmp_path / "train.jsonl"
    train.write_text(
        json.dumps({"id": "clean", "text": "unrelated short text"}) + "\n" +
        json.dumps({"id": "leaked", "text": "prefix " + phrase + " suffix"}) + "\n",
        encoding="utf-8",
    )
    result = _contamination_scan(train, benchmark, n=13)
    assert result["matching_documents"] == 1
    assert result["matches"][0]["document_id"] == "leaked"
