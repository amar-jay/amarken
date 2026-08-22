import json

import pytest

from src.distillation.grounded_pilot import audit_record, calculate, execute_tool, generate


def test_calculator_is_deterministic_and_restricted():
    assert calculate("(7+3)*2") == 20
    with pytest.raises(ValueError): calculate("2**8")
    with pytest.raises(ValueError): calculate("__import__('os')")


def test_tool_executor_uses_fixtures():
    assert execute_tool("calculator", {"expression":"13*4"})["result"] == 52
    assert execute_tool("get_weather", {"city":"Ankara","language":"tr"}) == {"city":"Ankara","condition":"güneşli","temperature_c":22}


def test_pilot_is_grounded_balanced_and_reproducible(tmp_path):
    keys = {"facts":[{"id":"x","en":{"question":"Q?","answer":"Paris.","required":["Paris"]},"tr":{"question":"S?","answer":"Paris.","required":["Paris"]}}]}
    keys_path=tmp_path/"keys.json"; keys_path.write_text(json.dumps(keys))
    config={"schema_version":1,"seed":7,"model":"teacher","base_url":"http://unused","answer_keys":str(keys_path),"contamination_references":[],"output_dir":str(tmp_path/"out"),"qa_per_language":1,"reasoning_per_language":2,"tools_per_language":2,"max_attempts":2,"max_answer_words":28,"validation_fraction":0.25}
    config_path=tmp_path/"config.json"; config_path.write_text(json.dumps(config))
    def fake(_url, _model, _lang, _user, verified, _attempt, _seed): return verified, {"test":True}
    first=generate(config_path,fake); second=generate(config_path,fake)
    assert first["requested"] == first["accepted"] == 10
    assert first["outputs"] == second["outputs"]
    records=[]; groups={}
    for split in ("train","validation"):
        split_records=[json.loads(line) for line in (tmp_path/f"out/{split}.jsonl").read_text().splitlines()]
        records += split_records; groups[split]={row["group"] for row in split_records}
    assert groups["train"].isdisjoint(groups["validation"])
    assert {row["category"] for row in records} == {"qa","reasoning","tool"}
    assert {row["language"] for row in records} == {"en","tr"}
    assert all("verified_target" in row["grounding"] for row in records)
    assert all(audit_record(row) == [] for row in records)
    tool_record=next(row for row in records if row["category"] == "tool")
    call=tool_record["messages"][2]["tool_calls"][0]["function"]
    assert execute_tool(call["name"],call["arguments"]) == tool_record["grounding"]["result"]
