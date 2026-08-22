"""Build and run a deterministic Ollama teacher qualification suite.

The suite measures the behavior Amarken actually needs: concise English and
Turkish answers, short arithmetic reasoning, native tool selection, and using
tool results.  It deliberately contains no programming or code-generation
tasks.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = ROOT / "benchmarks" / "ollama_teacher_qualification_v1.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "ollama_teacher_qwen3_5_2b.jsonl"
SYSTEM_PROMPT = (
    "You are a concise bilingual English/Turkish assistant. Answer only what was "
    "asked, normally in one short sentence. Do not explain unless explicitly asked. "
    "Use a provided tool only when external/current information or an action is needed."
)

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search for current information", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "create_reminder", "description": "Create a reminder", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "when": {"type": "string"}}, "required": ["text", "when"]}}},
]


EN_FACTS = [
    ("What is the capital of France? Answer briefly.", ["paris"]),
    ("Which planet is closest to the Sun? Answer briefly.", ["mercury"]),
    ("What gas do plants absorb from the atmosphere?", ["carbon dioxide"]),
    ("How many sides does a hexagon have?", ["6"]),
    ("At sea level, at what Celsius temperature does water freeze?", ["0"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("Which organ pumps blood through the body?", ["heart"]),
    ("What is the opposite of 'temporary'?", ["permanent"]),
    ("How many minutes are in two hours?", ["120"]),
    ("What currency is used in Japan?", ["yen"]),
    ("Which continent is Kenya in?", ["africa"]),
    ("What is H2O commonly called?", ["water"]),
    ("How many days are in a leap year?", ["366"]),
    ("What do bees produce?", ["honey"]),
    ("Which direction is opposite to east?", ["west"]),
    ("What is the square root of 81?", ["9"]),
    ("Which language is primarily spoken in Brazil?", ["portuguese"]),
    ("What is a young cat called?", ["kitten"]),
    ("How many centimeters are in a meter?", ["100"]),
    ("Which metal is liquid near room temperature?", ["mercury"]),
    ("What is the capital of Türkiye?", ["ankara"]),
    ("Which season follows spring?", ["summer"]),
    ("What is the largest mammal?", ["blue whale"]),
    ("How many vowels are in the English alphabet?", ["5"]),
    ("What force pulls objects toward Earth?", ["gravity"]),
]

TR_FACTS = [
    ("Türkiye'nin başkenti neresidir? Kısa cevap ver.", ["ankara"]),
    ("Güneş'e en yakın gezegen hangisidir?", ["merkür"]),
    ("Bitkiler atmosferden hangi gazı alır?", ["karbondioksit"]),
    ("Altıgenin kaç kenarı vardır?", ["6"]),
    ("Su deniz seviyesinde kaç santigrat derecede donar?", ["0"]),
    ("Dünyanın en büyük okyanusu hangisidir?", ["pasifik"]),
    ("Vücuda kan pompalayan organ hangisidir?", ["kalp"]),
    ("'Geçici' kelimesinin zıt anlamlısı nedir?", ["kalıcı"]),
    ("İki saat kaç dakikadır?", ["120"]),
    ("Japonya'nın para birimi nedir?", ["yen"]),
    ("Kenya hangi kıtadadır?", ["afrika"]),
    ("H2O'nun yaygın adı nedir?", ["su"]),
    ("Artık yılda kaç gün vardır?", ["366"]),
    ("Arılar ne üretir?", ["bal"]),
    ("Doğunun zıt yönü hangisidir?", ["batı"]),
    ("81'in karekökü kaçtır?", ["9"]),
    ("Brezilya'da ağırlıklı olarak hangi dil konuşulur?", ["portekizce"]),
    ("Yavru kediye ne denir?", ["yavru"]),
    ("Bir metrede kaç santimetre vardır?", ["100"]),
    ("Oda sıcaklığına yakın sıcaklıkta sıvı olan metal hangisidir?", ["cıva"]),
    ("Fransa'nın başkenti neresidir?", ["paris"]),
    ("İlkbahardan sonra hangi mevsim gelir?", ["yaz"]),
    ("En büyük memeli hangisidir?", ["mavi balina"]),
    ("Türk alfabesinde kaç harf vardır?", ["29"]),
    ("Cisimleri Dünya'ya çeken kuvvet nedir?", ["yerçekimi"]),
]


def build_suite() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for lang, facts in (("en", EN_FACTS), ("tr", TR_FACTS)):
        for repeat in range(2):
            for index, (prompt, keywords) in enumerate(facts):
                suffix = " Use one short sentence." if lang == "en" else " Tek kısa cümle kullan."
                cases.append({"id": f"qa-{lang}-{repeat * 25 + index:03d}", "category": f"qa_{lang}", "messages": [{"role": "user", "content": prompt + (suffix if repeat else "")}], "score": {"kind": "keywords", "all": keywords, "max_words": 24}})

    # Forty distinct, exact-answer reasoning cases; half in each language.
    for i in range(20):
        a, b, c = 7 + i, 3 + (i % 6), 2 + (i % 4)
        answer = (a + b) * c
        cases.append({"id": f"reason-en-{i:03d}", "category": "reasoning_en", "messages": [{"role": "user", "content": f"A box has {a} red balls and {b} blue balls. There are {c} identical boxes. How many balls are there total? Reply with only the number."}], "score": {"kind": "exact", "value": str(answer)}})
        cases.append({"id": f"reason-tr-{i:03d}", "category": "reasoning_tr", "messages": [{"role": "user", "content": f"Bir kutuda {a} kırmızı ve {b} mavi top var. Aynı kutudan {c} tane var. Toplam kaç top vardır? Yalnızca sayıyı yaz."}], "score": {"kind": "exact", "value": str(answer)}})

    cities = [("London", "Londra"), ("Berlin", "Berlin"), ("Ankara", "Ankara"), ("Tokyo", "Tokyo"), ("Paris", "Paris")]
    for i in range(20):
        en = i % 2 == 0
        city = cities[i % len(cities)][0 if en else 1]
        prompt = (f"What is the weather in {city} right now?" if en else f"{city} için şu an hava nasıl?")
        cases.append({"id": f"tool-needed-{i:03d}", "category": "tool_routing", "tools": TOOLS, "messages": [{"role": "user", "content": prompt}], "score": {"kind": "tool", "name": "get_weather", "arguments": {"city": city}}})
    direct_prompts = [("en", "What is 2 + 2? Answer directly.", ["4"]), ("tr", "2 + 2 kaçtır? Doğrudan cevap ver.", ["4"]), ("en", "Say hello in one word.", ["hello"]), ("tr", "Tek kelimeyle merhaba de.", ["merhaba"])]
    for i in range(20):
        lang, prompt, words = direct_prompts[i % len(direct_prompts)]
        cases.append({"id": f"tool-direct-{i:03d}", "category": "tool_routing", "tools": TOOLS, "messages": [{"role": "user", "content": prompt}], "score": {"kind": "keywords", "all": words, "max_words": 12}})

    for i in range(20):
        tr = i % 2 == 1
        city = cities[i % len(cities)][1 if tr else 0]
        condition, temp = (("güneşli", "22") if tr else ("sunny", "22"))
        cases.append({"id": f"tool-result-{i:03d}", "category": "tool_result", "tools": TOOLS, "messages": [
            {"role": "user", "content": (f"{city} için hava nasıl?" if tr else f"What is the weather in {city}?")},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": city}}}]},
            {"role": "tool", "tool_name": "get_weather", "content": json.dumps({"city": city, "condition": condition, "temperature_c": 22}, ensure_ascii=False)},
        ], "score": {"kind": "keywords", "all": [condition, temp], "max_words": 28}})

    assert len(cases) == 200
    return {"name": "ollama-teacher-qualification-v1", "frozen": True, "case_count": len(cases), "cases": cases}


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"[^\wçğıöşü]+", " ", text).strip()
    number_words = {"zero": "0", "six": "6", "nine": "9", "five": "5", "sıfır": "0", "altı": "6", "dokuz": "9", "beş": "5"}
    return " ".join(number_words.get(token, token) for token in text.split())


def score_case(case: dict[str, Any], message: dict[str, Any]) -> tuple[bool, str]:
    spec, calls = case["score"], message.get("tool_calls") or []
    if spec["kind"] == "tool":
        if len(calls) != 1:
            return False, f"expected one tool call, got {len(calls)}"
        fn = calls[0].get("function", {})
        if fn.get("name") != spec["name"]:
            return False, f"wrong tool: {fn.get('name')}"
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try: arguments = json.loads(arguments)
            except json.JSONDecodeError: return False, "invalid argument JSON"
        for key, value in spec["arguments"].items():
            if normalize(arguments.get(key, "")) != normalize(value):
                return False, f"wrong argument: {key}"
        return True, "pass"
    if calls:
        return False, "unnecessary tool call"
    content = message.get("content") or ""
    if spec["kind"] == "exact":
        return (normalize(content) == normalize(spec["value"]), "pass" if normalize(content) == normalize(spec["value"]) else "not exact")
    haystack = normalize(content)
    if not all(normalize(word) in haystack for word in spec["all"]):
        return False, "missing required content"
    if len(content.split()) > spec["max_words"]:
        return False, "too verbose"
    return True, "pass"


def chat(base_url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(base_url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run(args: argparse.Namespace) -> None:
    suite = json.loads(args.suite.read_text())
    done: dict[str, dict[str, Any]] = {}
    if args.output.exists() and not args.restart:
        for line in args.output.read_text().splitlines():
            row = json.loads(line); done[row["id"]] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.restart else "a"
    with args.output.open(mode) as output:
        selected = suite["cases"][: args.limit] if args.limit else suite["cases"]
        for number, case in enumerate(selected, 1):
            if case["id"] in done: continue
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *case["messages"]]
            payload = {"model": args.model, "messages": messages, "stream": False, "think": False, "options": {"seed": args.seed, "temperature": 0, "num_ctx": 2048, "num_predict": 48}}
            if case.get("tools"): payload["tools"] = case["tools"]
            started = time.monotonic()
            try:
                response = chat(args.base_url, payload, args.timeout)
                passed, reason = score_case(case, response.get("message", {}))
                row = {"id": case["id"], "category": case["category"], "pass": passed, "reason": reason, "elapsed_s": round(time.monotonic() - started, 3), "message": response.get("message"), "eval_count": response.get("eval_count"), "eval_duration": response.get("eval_duration")}
            except Exception as exc:
                row = {"id": case["id"], "category": case["category"], "pass": False, "reason": f"request error: {exc}", "elapsed_s": round(time.monotonic() - started, 3)}
            output.write(json.dumps(row, ensure_ascii=False) + "\n"); output.flush(); done[case["id"]] = row
            print(f"[{number:03d}/{len(selected):03d}] {case['id']}: {'PASS' if row['pass'] else 'FAIL'} ({row['reason']})")
    counts, passes = Counter(), Counter()
    for row in done.values(): counts[row["category"]] += 1; passes[row["category"]] += int(row["pass"])
    summary = {key: {"passed": passes[key], "total": counts[key], "rate": round(passes[key] / counts[key], 4)} for key in sorted(counts)}
    summary["overall"] = {"passed": sum(passes.values()), "total": sum(counts.values()), "rate": round(sum(passes.values()) / sum(counts.values()), 4)}
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"model": args.model, "suite": suite["name"], "results": summary}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build"); build.add_argument("--output", type=Path, default=DEFAULT_SUITE)
    qualify = sub.add_parser("run"); qualify.add_argument("--suite", type=Path, default=DEFAULT_SUITE); qualify.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); qualify.add_argument("--model", default="qwen3.5:2b-q4_K_M"); qualify.add_argument("--base-url", default="http://127.0.0.1:11434"); qualify.add_argument("--seed", type=int, default=3407); qualify.add_argument("--timeout", type=int, default=180); qualify.add_argument("--limit", type=int); qualify.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(build_suite(), indent=2, ensure_ascii=False) + "\n"); print(args.output)
    else: run(args)


if __name__ == "__main__": main()
