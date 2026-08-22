from src.distillation.ollama_qualification import build_suite, score_case


def test_suite_is_frozen_and_balanced():
    suite = build_suite()
    assert suite["frozen"] is True
    assert suite["case_count"] == 200
    assert len({case["id"] for case in suite["cases"]}) == 200
    counts = {}
    for case in suite["cases"]:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    assert counts == {"qa_en": 50, "qa_tr": 50, "reasoning_en": 20, "reasoning_tr": 20, "tool_routing": 40, "tool_result": 20}


def test_tool_and_direct_scoring():
    case = build_suite()["cases"][140]
    ok, _ = score_case(case, {"tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "London"}}}]})
    assert ok
    direct = build_suite()["cases"][160]
    ok, _ = score_case(direct, {"content": "4", "tool_calls": []})
    assert ok
