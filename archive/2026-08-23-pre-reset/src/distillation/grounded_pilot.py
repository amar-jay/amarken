"""necessities for Grounding EN/TR assistant SFT pilot.

Qwen is a surface realizer only. Facts come from hash-bound answer keys,
reasoning answers from deterministic arithmetic, and tool results from local
executors. An output is accepted only if it preserves its independently created
target and passes structural checks.
"""

from __future__ import annotations

import ast
import hashlib
import json
import operator
import re
import urllib.request
from pathlib import Path
from typing import Any, Callable

import random
from collections import Counter



ROOT = Path(__file__).resolve().parents[2]
ASSISTANT_SYSTEM = "You are a concise, helpful English and Turkish conversational assistant. Do not produce programming code."
GENERATOR_SYSTEM = "Reply as the assistant to the conversation. Use the requested language, be concise and natural, and output only the answer. Never output programming code or markdown code blocks."

# perhaps for tool calling later, but not for answering
OPS: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
}

def valid(text: str, spec: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    words = text.split()
    if len(words) < config["min_words"]:
        return False, "too_short"
    if len(words) > config["max_words"]:
        return False, "too_long"
    numeric_text = re.sub(r"(?<=\d)[ ,.](?=\d)", "", text)
    if any(str(value) not in text and str(value) not in numeric_text for value in spec["required"]):
        return False, "required_value_missing"
    if "```" in text or re.search(
        r"\b(import|def|class|function|javascript|python|html|css)\b", text, re.I
    ):
        return False, "code_like"
    if re.search(r"\b(as an ai|bir yapay zeka olarak)\b", text):
        return False, "model_disclaimer"
    tokens = text.split()
    if tokens and Counter(tokens).most_common(1)[0][1] / len(tokens) > 0.42:
        return False, "token_repetition"
    tr_score = sum(token in TR_MARKERS for token in tokens) + sum(
        char in text for char in "çğıöşüÇĞİÖŞÜ"
    )
    if spec["language"] == "tr" and tr_score < 1:
        return False, "wrong_language"
    if spec["language"] == "bilingual" and tr_score < 1:
        return False, "missing_turkish"
    return True, "accepted"



def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate(expression: str) -> int:
    node = ast.parse(expression, mode="eval").body

    def visit(item: ast.AST) -> int:
        if (
            isinstance(item, ast.Constant)
            and type(item.value) is int
            and abs(item.value) <= 10_000
        ):
            return item.value
        if isinstance(item, ast.BinOp) and type(item.op) in OPS:
            return OPS[type(item.op)](visit(item.left), visit(item.right))
        raise ValueError("unsupported calculator expression")

    result = visit(node)
    if abs(result) > 1_000_000:
        raise ValueError("calculator result out of range")
    return result


def ollama_json(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 180,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def model_identity(base_url: str, model: str) -> dict[str, Any]:
    tags = ollama_json(base_url, "/api/tags")
    for item in tags.get("models", []):
        if item.get("name") == model or item.get("model") == model:
            return {
                key: item.get(key)
                for key in ("name", "model", "digest", "size", "modified_at")
            }
    raise RuntimeError(f"model not found in Ollama: {model}")


def valid_answer(text: str, required: list[str], max_words: int) -> tuple[bool, str]:
    if not text:
        return False, "empty"
    if len(text.split()) > max_words:
        return False, "too_verbose"
    haystack = text
    if not all(value in haystack for value in required):
        return False, "target_not_preserved"
    if "```" in text or re.search(r"\b(import|def|function|class)\b", text, re.I):
        return False, "code_like_output"
    return True, "accepted"


def split_for(group: str, fraction: float) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "validation" if bucket < fraction else "train"


def recursive_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from recursive_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from recursive_strings(item)


def contamination_index(paths: list[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    values, manifest = set(), []
    for path in paths:
        data = json.loads(path.read_text())
        values.update(
            text
            for text in recursive_strings(data)
            if len(text.split()) >= 3
        )
        manifest.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    return values, manifest


def hash_fraction(value: str) -> float:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def language_for(index: int) -> str:
    slot = index % 20
    return "en" if slot < 9 else "tr" if slot < 18 else "bilingual"


def category_for(index: int) -> str:
    slot = index % 100
    if slot < 30:
        return "conversation"
    if slot < 50:
        return "contextual_qa"
    if slot < 65:
        return "transformation"
    if slot < 80:
        return "reasoning"
    if slot < 90:
        return "general_qa"
    return "multi_turn"


NAMES = [
    "Ada",
    "Mert",
    "Elif",
    "Deniz",
    "Ece",
    "Can",
    "Maya",
    "Noah",
    "Lina",
    "Arda",
    "Zeynep",
    "Emir",
]
PLACES = [
    "Ankara",
    "Istanbul",
    "Izmir",
    "Antalya",
    "London",
    "Berlin",
    "Lisbon",
    "Oslo",
    "Vienna",
    "Prague",
]
OBJECTS = [
    "notebook",
    "backpack",
    "tea set",
    "bicycle",
    "lamp",
    "plant",
    "book",
    "ticket",
    "camera",
    "jacket",
]
ACTIVITIES = [
    "reading",
    "walking",
    "cooking",
    "gardening",
    "painting",
    "journaling",
    "swimming",
    "photography",
    "yoga",
    "learning a language",
]
TR_ACTIVITIES = [
    "kitap okumak",
    "yürüyüş yapmak",
    "yemek pişirmek",
    "bahçeyle ilgilenmek",
    "resim yapmak",
    "günlük tutmak",
    "yüzmek",
    "fotoğraf çekmek",
    "yoga yapmak",
    "dil öğrenmek",
]
TONES = ["friendly", "calm", "encouraging", "direct", "warm", "practical"]
TOPICS = [
    "habits",
    "friendship",
    "motivation",
    "travel planning",
    "sleep routine",
    "time management",
    "home organization",
    "communication",
    "learning",
    "well-being",
]
TR_TOPICS = [
    "alışkanlıklar",
    "arkadaşlık",
    "motivasyon",
    "seyahat planı",
    "uyku düzeni",
    "zaman yönetimi",
    "ev düzeni",
    "iletişim",
    "öğrenme",
    "iyi yaşam",
]
TR_MARKERS = {
    "bir",
    "ve",
    "için",
    "nasıl",
    "nedir",
    "var",
    "olarak",
    "bana",
    "kısa",
    "lütfen",
    "olur",
    "ile",
    "bu",
    "çok",
    "daha",
}
def make_spec(index: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed + index * 1_000_003)
    lang = language_for(index)
    category = category_for(index)
    name, other = rng.sample(NAMES, 2)
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    number = 10_000 + index
    required: list[str] = []
    if category == "conversation":
        if lang == "tr":
            user = f"{rng.choice(TR_TOPICS)} konusunda zorlanan birine iki kısa ve uygulanabilir öneri ver. Durum numarası {number}."
        elif lang == "bilingual":
            user = f"Give one short suggestion about {rng.choice(TOPICS)} in English, then repeat it naturally in Turkish. Scenario {number}."
        else:
            user = f"Give two short, practical suggestions to someone struggling with {rng.choice(TOPICS)}. Scenario {number}."
    elif category == "contextual_qa":
        count = 2 + rng.randrange(8)
        color = rng.choice(["red", "blue", "green", "yellow", "white"])
        tr_color = {
            "red": "kırmızı",
            "blue": "mavi",
            "green": "yeşil",
            "yellow": "sarı",
            "white": "beyaz",
        }[color]
        if lang == "tr":
            user = f"{number} numaralı kayıt: {name}, {place} kentindeki {tr_color} {obj} için {count} adet ayırdı. {name} kaç adet ayırdı?"
            required = [str(count)]
        elif lang == "bilingual":
            user = f"Record {number}: {name} reserved {count} {color} items in {place}. Answer the quantity first in English and then in Turkish."
            required = [str(count)]
        else:
            user = f"Record {number}: {name} reserved {count} {color} {obj}s in {place}. How many did {name} reserve?"
            required = [str(count)]
    elif category == "transformation":
        activity = rng.choice(TR_ACTIVITIES if lang == "tr" else ACTIVITIES)
        day = 1 + rng.randrange(28)
        source = (
            f"{name}, {activity} için ayın {day}. gününde {place} kentinde {other} ile buluşacak."
            if lang == "tr"
            else f"On day {day}, {name} will meet {other} in {place} for {activity}."
        )
        if lang == "bilingual":
            source = f"On day {day}, {name} will meet {other} in {place}."
            user = f"Summarize this in one short English sentence and one Turkish sentence: {source}"
        else:
            user = (
                "Bu cümleyi anlamını değiştirmeden daha kısa yaz: "
                if lang == "tr"
                else "Rewrite this more concisely without changing its meaning: "
            ) + source
        required = [name, place]
    elif category == "reasoning":
        a = 3 + (index * 17) % 997
        b = 2 + (index * 31) % 211
        c = 2 + (index * 7) % 8
        result = calculate(f"({a}+{b})*{c}")
        if lang == "tr":
            user = f"Her kutuda {a} kalem ve {b} silgi var. {c} kutuda toplam kaç eşya vardır? Yalnızca kısa cevap ver."
            required = [str(result)]
        elif lang == "bilingual":
            user = f"Each box has {a} pencils and {b} erasers. There are {c} boxes. Give the total briefly in English and Turkish."
            required = [str(result)]
        else:
            user = f"Each box has {a} pencils and {b} erasers. How many items are in {c} boxes? Answer briefly."
            required = [str(result)]
    elif category == "general_qa":
        topic = rng.choice(TR_TOPICS if lang == "tr" else TOPICS)
        if lang == "tr":
            user = f"{topic} hakkında yeni başlayan birinin sorabileceği basit bir soruyu kısa ve yararlı biçimde yanıtla. Örnek {number}."
        elif lang == "bilingual":
            user = f"Answer a beginner's everyday question about {rng.choice(TOPICS)} briefly in English and Turkish. Example {number}."
        else:
            user = f"Briefly answer a simple beginner's everyday question about {topic}. Example {number}."
    else:
        earlier = (
            f"Son zamanlarda {rng.choice(TR_ACTIVITIES)} istiyorum."
            if lang == "tr"
            else f"I want to start {rng.choice(ACTIVITIES)} soon."
        )
        reply = (
            "Bunu küçük bir adımla başlatabiliriz."
            if lang == "tr"
            else "We can start with one small step."
        )
        if lang == "bilingual":
            earlier = f"I want to start {rng.choice(ACTIVITIES)} soon."
            reply = "Let's make it manageable."
            user = "Suggest the first step in English, then Turkish."
        else:
            user = (
                "İlk adımım ne olsun?"
                if lang == "tr"
                else "What should my first step be?"
            )
        return {
            "id": f"syn-{index:09d}",
            "group": f"syn-{index:09d}",
            "category": category,
            "language": lang,
            "messages": [
                {"role": "system", "content": ASSISTANT_SYSTEM},
                {"role": "user", "content": earlier},
                {"role": "assistant", "content": reply},
                {"role": "user", "content": user},
            ],
            "required": required,
        }
    return {
        "id": f"syn-{index:09d}",
        "group": f"syn-{index:09d}",
        "category": category,
        "language": lang,
        "messages": [
            {"role": "system", "content": ASSISTANT_SYSTEM},
            {"role": "user", "content": user},
        ],
        "required": required,
    }
