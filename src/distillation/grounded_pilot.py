"""Generate a small, grounded EN/TR assistant SFT pilot through Ollama.

Qwen is a surface realizer only. Facts come from hash-bound answer keys,
reasoning answers from deterministic arithmetic, and tool results from local
executors. An output is accepted only if it preserves its independently created
target and passes structural checks.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import operator
import random
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = "You are a concise bilingual English/Turkish assistant. Obey the requested format exactly. Do not add unsupported facts."
WEATHER = {
    "Ankara": {"condition": {"en": "sunny", "tr": "güneşli"}, "temperature_c": 22},
    "Istanbul": {"condition": {"en": "cloudy", "tr": "bulutlu"}, "temperature_c": 18},
    "London": {"condition": {"en": "rainy", "tr": "yağmurlu"}, "temperature_c": 14},
    "Berlin": {"condition": {"en": "windy", "tr": "rüzgarlı"}, "temperature_c": 16},
}
TOOL_SCHEMAS = [
    {"type":"function","function":{"name":"get_weather","description":"Get weather from the local fixture","parameters":{"type":"object","properties":{"city":{"type":"string"},"language":{"type":"string","enum":["en","tr"]}},"required":["city","language"]}}},
    {"type":"function","function":{"name":"calculator","description":"Evaluate basic arithmetic","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}},
]
OPS: dict[type[ast.operator], Callable[[int, int], int]] = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def canonical(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).replace("İ", "I").casefold()
    text = "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))
    return re.sub(r"[^\wçğıöşü]+", " ", text).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def calculate(expression: str) -> int:
    node = ast.parse(expression, mode="eval").body
    def visit(item: ast.AST) -> int:
        if isinstance(item, ast.Constant) and type(item.value) is int and abs(item.value) <= 10_000: return item.value
        if isinstance(item, ast.BinOp) and type(item.op) in OPS: return OPS[type(item.op)](visit(item.left), visit(item.right))
        raise ValueError("unsupported calculator expression")
    result = visit(node)
    if abs(result) > 1_000_000: raise ValueError("calculator result out of range")
    return result


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "calculator":
        expression = str(arguments["expression"])
        return {"expression": expression, "result": calculate(expression)}
    if name == "get_weather":
        city, language = str(arguments["city"]), str(arguments["language"])
        fixture = WEATHER[city]
        return {"city": city, "condition": fixture["condition"][language], "temperature_c": fixture["temperature_c"]}
    raise ValueError(f"unknown tool {name}")


def ollama_json(base_url: str, endpoint: str, payload: dict[str, Any] | None = None, timeout: int = 180) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base_url.rstrip("/") + endpoint, data=data, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response: return json.load(response)


def model_identity(base_url: str, model: str) -> dict[str, Any]:
    tags = ollama_json(base_url, "/api/tags")
    for item in tags.get("models", []):
        if item.get("name") == model or item.get("model") == model:
            return {key: item.get(key) for key in ("name", "model", "digest", "size", "modified_at")}
    raise RuntimeError(f"model not found in Ollama: {model}")


def phrase(base_url: str, model: str, language: str, user: str, verified: str, attempt: int, seed: int) -> tuple[str, dict[str, Any]]:
    lang = "English" if language == "en" else "Turkish"
    instruction = f"Write only one concise {lang} assistant answer to the user. Preserve this verified target exactly in meaning and values: {verified!r}. Do not solve or change it."
    if attempt: instruction += " Previous output was rejected; copy every named entity and number from the verified target."
    payload = {"model":model,"stream":False,"think":False,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":instruction + "\nUser: " + user}],"options":{"seed":seed + attempt,"temperature":0,"num_ctx":2048,"num_predict":64},"keep_alive":"10m"}
    response = ollama_json(base_url, "/api/chat", payload)
    return response.get("message", {}).get("content", "").strip(), {key: response.get(key) for key in ("model","created_at","done_reason","total_duration","prompt_eval_count","eval_count")}


def valid_answer(text: str, required: list[str], max_words: int) -> tuple[bool, str]:
    if not text: return False, "empty"
    if len(text.split()) > max_words: return False, "too_verbose"
    haystack = canonical(text)
    if not all(canonical(value) in haystack for value in required): return False, "target_not_preserved"
    if "```" in text or re.search(r"\b(import|def|function|class)\b", text, re.I): return False, "code_like_output"
    return True, "accepted"


def specs(config: dict[str, Any], keys: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fact in keys["facts"][: config["qa_per_language"]]:
        for lang in ("en", "tr"):
            item = fact[lang]
            result.append({"id":f"qa-{lang}-{fact['id']}","group":f"qa-{fact['id']}","category":"qa","language":lang,"user":item["question"],"verified":item["answer"],"required":item["required"],"truth":{"kind":"answer_key","key_id":fact["id"]}})
    for lang in ("en", "tr"):
        for index in range(config["reasoning_per_language"]):
            a, b, count = 6 + index, 2 + index % 5, 2 + index % 4
            expression, value = f"({a}+{b})*{count}", calculate(f"({a}+{b})*{count}")
            user = (f"Each basket has {a} apples and {b} pears. There are {count} baskets. How many fruits are there in total?" if lang == "en" else f"Her sepette {a} elma ve {b} armut var. {count} sepet olduğuna göre toplam kaç meyve vardır?")
            verified = (f"There are {value} fruits in total." if lang == "en" else f"Toplam {value} meyve vardır.")
            result.append({"id":f"reason-{lang}-{index:03d}","group":f"reason-{index:03d}","category":"reasoning","language":lang,"user":user,"verified":verified,"required":[str(value)],"truth":{"kind":"calculator","expression":expression,"result":value}})
    cities = list(WEATHER)
    for lang in ("en", "tr"):
        for index in range(config["tools_per_language"]):
            if index % 2 == 0:
                city = cities[(index // 2) % len(cities)]
                args = {"city":city,"language":lang}; tool = "get_weather"; tool_result = execute_tool(tool, args)
                user = f"What is the fixture weather in {city}?" if lang == "en" else f"{city} için kayıtlı hava durumu nedir?"
                condition, temp = tool_result["condition"], tool_result["temperature_c"]
                verified = f"{city} is {condition} at {temp}°C." if lang == "en" else f"{city} {temp}°C ve {condition}."
                required = [city, condition, str(temp)]
            else:
                a, b = 13 + index, 4 + index % 6; expression = f"{a}*{b}"; tool = "calculator"; args = {"expression":expression}; tool_result = execute_tool(tool, args)
                user = f"Calculate {expression}." if lang == "en" else f"{expression} işlemini hesapla."
                verified = f"The result is {tool_result['result']}." if lang == "en" else f"Sonuç {tool_result['result']}."
                required = [str(tool_result["result"])]
            result.append({"id":f"tool-{lang}-{index:03d}","group":f"tool-{index:03d}","category":"tool","language":lang,"user":user,"verified":verified,"required":required,"truth":{"kind":"executed_tool","tool":tool,"arguments":args,"result":tool_result}})
    return result


def split_for(group: str, fraction: float) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "validation" if bucket < fraction else "train"


def build_record(spec: dict[str, Any], answer: str, generator: dict[str, Any], keys_hash: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role":"system","content":SYSTEM},{"role":"user","content":spec["user"]}]
    record: dict[str, Any] = {"id":spec["id"],"group":spec["group"],"category":spec["category"],"language":spec["language"],"messages":messages,"grounding":{"answer_keys_sha256":keys_hash,**spec["truth"],"verified_target":spec["verified"],"required":spec["required"]},"generator":generator}
    if spec["category"] == "tool":
        truth = spec["truth"]
        messages.extend([{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":truth["tool"],"arguments":truth["arguments"]}}]},{"role":"tool","tool_name":truth["tool"],"content":json.dumps(truth["result"],ensure_ascii=False)},{"role":"assistant","content":answer}])
        record["tools"] = TOOL_SCHEMAS
    else: messages.append({"role":"assistant","content":answer})
    return record


def recursive_strings(value: Any):
    if isinstance(value, str): yield value
    elif isinstance(value, list):
        for item in value: yield from recursive_strings(item)
    elif isinstance(value, dict):
        for item in value.values(): yield from recursive_strings(item)


def contamination_index(paths: list[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    values, manifest = set(), []
    for path in paths:
        data = json.loads(path.read_text())
        values.update(canonical(text) for text in recursive_strings(data) if len(text.split()) >= 3)
        manifest.append({"path":str(path.relative_to(ROOT)),"sha256":sha256(path)})
    return values, manifest


def audit_record(record: dict[str, Any], benchmark_strings: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    if record.get("category") not in {"qa","reasoning","tool"}: errors.append("invalid_or_code_category")
    if record.get("language") not in {"en","tr"}: errors.append("invalid_language")
    messages = record.get("messages", [])
    if len(messages) < 3 or messages[1].get("role") != "user" or messages[-1].get("role") != "assistant": errors.append("invalid_messages")
    if benchmark_strings is not None and len(messages) > 1 and canonical(messages[1].get("content", "")) in benchmark_strings: errors.append("benchmark_exact_overlap")
    grounding = record.get("grounding", {}); answer = messages[-1].get("content", "") if messages else ""
    ok, reason = valid_answer(answer, grounding.get("required", []), 28)
    if not ok: errors.append(reason)
    if record.get("category") == "reasoning":
        try:
            if calculate(grounding["expression"]) != grounding["result"]: errors.append("reasoning_truth_mismatch")
        except (KeyError, ValueError): errors.append("invalid_reasoning_truth")
    if record.get("category") == "tool":
        try:
            call = messages[2]["tool_calls"][0]["function"]
            executed = execute_tool(call["name"], call["arguments"])
            if call["name"] != grounding["tool"] or executed != grounding["result"]: errors.append("tool_replay_mismatch")
            if json.loads(messages[3]["content"]) != executed: errors.append("tool_message_mismatch")
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError): errors.append("invalid_tool_transcript")
    return errors


def generate(config_path: Path, phrase_fn: Callable[..., tuple[str, dict[str, Any]]] = phrase) -> dict[str, Any]:
    config = json.loads(config_path.read_text()); keys_path = ROOT / config["answer_keys"]; keys = json.loads(keys_path.read_text()); output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True); identity = model_identity(config["base_url"], config["model"]) if phrase_fn is phrase else {"name":config["model"],"digest":"test-double"}
    reference_paths = [ROOT / path for path in config.get("contamination_references", [])]
    benchmark_strings, reference_manifest = contamination_index(reference_paths)
    rows, rejects = {"train":[],"validation":[]}, []
    for index, spec in enumerate(specs(config, keys)):
        if canonical(spec["user"]) in benchmark_strings:
            rejects.append({"id":spec["id"],"reason":"benchmark_exact_overlap"}); continue
        accepted = None
        for attempt in range(config["max_attempts"]):
            text, metrics = phrase_fn(config["base_url"], config["model"], spec["language"], spec["user"], spec["verified"], attempt, config["seed"] + index)
            ok, reason = valid_answer(text, spec["required"], config["max_answer_words"])
            if ok:
                candidate = build_record(spec, text, {"model":identity,"options":{"seed":config["seed"]+index+attempt,"temperature":0,"think":False,"num_ctx":2048,"num_predict":64},"attempt":attempt,"metrics":metrics}, sha256(keys_path))
                audit_errors = audit_record(candidate, benchmark_strings)
                if not audit_errors: accepted = candidate; break
                reason = ",".join(audit_errors)
            rejects.append({"id":spec["id"],"attempt":attempt,"reason":reason,"output":text})
        if accepted is not None: rows[split_for(spec["group"], config["validation_fraction"])].append(accepted)
        else: rejects.append({"id":spec["id"],"reason":"exhausted","verified_target":spec["verified"]})
    paths = {}
    for name, content in (("train",rows["train"]),("validation",rows["validation"]),("rejections",rejects)):
        path = output_dir / f"{name}.jsonl"; path.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in content)); paths[path.name] = {"lines":len(content),"bytes":path.stat().st_size,"sha256":sha256(path)}
    accepted = rows["train"] + rows["validation"]; counts = Counter((row["category"],row["language"]) for row in accepted)
    audit_errors = {row["id"]:errors for row in accepted if (errors:=audit_record(row, benchmark_strings))}
    manifest = {"schema_version":1,"config":config,"config_sha256":sha256(config_path),"answer_keys_sha256":sha256(keys_path),"contamination_references":reference_manifest,"model":identity,"requested":len(specs(config,keys)),"accepted":len(accepted),"rejected_specs":len(specs(config,keys))-len(accepted),"post_generation_audit":{"passed":len(accepted)-len(audit_errors),"failed":len(audit_errors),"errors":audit_errors},"counts":{f"{a}_{b}":n for (a,b),n in sorted(counts.items())},"outputs":paths}
    manifest_path = output_dir / "manifest.json"; manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    return manifest


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=ROOT/"configs/grounded_pilot_v1.json"); args=parser.parse_args(); print(json.dumps(generate(args.config),indent=2,ensure_ascii=False))


if __name__ == "__main__": main()
