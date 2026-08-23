"""Build the deterministic, bilingual meaningful-scale development benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

LETTERS = "ABCD"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _options(answer: str, distractors: list[str], position: int) -> list[str]:
    values = list(distractors[:3])
    values.insert(position, answer)
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError("benchmark options must contain four distinct values")
    return values


def _mc_tasks(seed: int) -> list[dict]:
    rng = random.Random(seed)
    tasks = []
    names = [("Mira", "Jon", "Ada", "Lina"), ("Ece", "Can", "Bora", "Duru")]
    colors = [
        ("amber", "cobalt", "silver", "violet"),
        ("kehribar", "kobalt", "gümüş", "mor"),
    ]
    for language in ("en", "tr"):
        for index in range(24):
            position = len(tasks) % 4
            a, b = rng.randint(3, 30), rng.randint(2, 15)
            answer = str((a + b) * 2)
            prompt = (
                f"Add {a} and {b}, then double the result."
                if language == "en"
                else f"{a} ile {b} sayısını topla, sonra sonucu ikiyle çarp."
            )
            tasks.append(
                {
                    "id": f"{language}-cr-{index:02d}",
                    "language": language,
                    "category": "compositional_reasoning",
                    "prompt": prompt,
                    "options": _options(
                        answer,
                        [str(a + b), str((a + b) * 2 - 1), str((a + b) * 2 + 1)],
                        position,
                    ),
                    "answer": answer,
                }
            )
        for index in range(24):
            position = len(tasks) % 4
            words = colors[0 if language == "en" else 1]
            requested = index % 4
            prompt = (
                f"Return the {('first','second','third','fourth')[requested]} word, not an alphabetic ordering: {' '.join(words)}."
                if language == "en"
                else f"Alfabetik sıralama yapmadan {('birinci','ikinci','üçüncü','dördüncü')[requested]} sözcüğü döndür: {' '.join(words)}."
            )
            tasks.append(
                {
                    "id": f"{language}-if-{index:02d}",
                    "language": language,
                    "category": "instruction_following",
                    "prompt": prompt,
                    "options": _options(
                        words[requested],
                        [word for word in words if word != words[requested]],
                        position,
                    ),
                    "answer": words[requested],
                }
            )
        for index in range(24):
            position = len(tasks) % 4
            codes = [
                f"Q{rng.randint(10,99)}",
                f"R{rng.randint(10,99)}",
                f"S{rng.randint(10,99)}",
            ]
            keys = ["Luma", "Taro", "Nira"]
            requested = index % 3
            reference = "; ".join(f"{key}={code}" for key, code in zip(keys, codes))
            prompt = (
                f"Reference: {reference}. {keys[requested]}?"
                if language == "en"
                else f"Kaynak: {reference}. {keys[requested]}?"
            )
            wrong = [code for code in codes if code != codes[requested]] + [
                codes[requested][::-1]
            ]
            tasks.append(
                {
                    "id": f"{language}-rt-{index:02d}",
                    "language": language,
                    "category": "retrieval",
                    "prompt": prompt,
                    "options": _options(codes[requested], wrong, position),
                    "answer": codes[requested],
                }
            )
        for index in range(24):
            position = len(tasks) % 4
            people = names[0 if language == "en" else 1]
            hops = 2 + index % 2
            chain = people[: hops + 1]
            if language == "en":
                prompt = (
                    f"{chain[0]} holds the key. "
                    + " ".join(
                        f"{left} gives it to {right}."
                        for left, right in zip(chain, chain[1:])
                    )
                    + " Who holds it now?"
                )
            else:
                prompt = (
                    f"Anahtar {chain[0]} kişisindedir. "
                    + " ".join(
                        f"{left} anahtarı {right} kişisine verir."
                        for left, right in zip(chain, chain[1:])
                    )
                    + " Anahtar şimdi kimde?"
                )
            tasks.append(
                {
                    "id": f"{language}-st-{index:02d}",
                    "language": language,
                    "category": "state_tracking",
                    "prompt": prompt,
                    "options": _options(
                        chain[-1],
                        [person for person in people if person != chain[-1]],
                        position,
                    ),
                    "answer": chain[-1],
                }
            )
        for index in range(24):
            position = len(tasks) % 4
            key, value = f"k{index}", rng.randint(1, 99)
            answer = json.dumps({key: value}, separators=(",", ":"))
            prompt = (
                f"Return the valid JSON object mapping {key} to numeric value {value}."
                if language == "en"
                else f"{key} anahtarını {value} sayısına eşleyen geçerli JSON nesnesini döndür."
            )
            distractors = [
                f"{{{key}:{value}}}",
                json.dumps({key: str(value)}, separators=(",", ":")),
                f"{key}={value}",
            ]
            tasks.append(
                {
                    "id": f"{language}-ts-{index:02d}",
                    "language": language,
                    "category": "tool_syntax",
                    "prompt": prompt,
                    "options": _options(answer, distractors, position),
                    "answer": answer,
                }
            )
    return tasks


def _generative_tasks(seed: int) -> list[dict]:
    rng = random.Random(seed + 1)
    tasks = []
    for language in ("en", "tr"):
        for index in range(30):
            a, b = rng.randint(2, 40), rng.randint(2, 20)
            prompt = (
                f"Calculate {a} + {b}. Answer with digits only:"
                if language == "en"
                else f"{a} + {b} işlemini hesapla. Yalnızca rakamlarla yanıtla:"
            )
            tasks.append(
                {
                    "id": f"{language}-gen-math-{index:02d}",
                    "language": language,
                    "category": "arithmetic_exact",
                    "prompt": prompt,
                    "answers": [str(a + b)],
                    "max_new_tokens": 4,
                }
            )
        for index in range(10):
            word = (
                ("cobalt", "silver", "amber", "violet")[index % 4]
                if language == "en"
                else ("kobalt", "gümüş", "kehribar", "mor")[index % 4]
            )
            prompt = (
                f"Repeat exactly this word: {word}\nAnswer:"
                if language == "en"
                else f"Bu sözcüğü aynen tekrarla: {word}\nYanıt:"
            )
            tasks.append(
                {
                    "id": f"{language}-gen-copy-{index:02d}",
                    "language": language,
                    "category": "copy_exact",
                    "prompt": prompt,
                    "answers": [word],
                    "max_new_tokens": 8,
                }
            )
    for index in range(40):
        name = f"value_{index}"
        prompt = f"Complete the Python expression with the identifier only.\n{name} = {index}\nresult = "
        tasks.append(
            {
                "id": f"code-gen-{index:02d}",
                "language": "code",
                "category": "code_completion",
                "prompt": prompt,
                "answers": [name],
                "max_new_tokens": 12,
            }
        )
    return tasks


def _lm_probes(validation_path: Path, seed: int, per_language: int = 30) -> list[dict]:
    rng = random.Random(seed + 2)
    reservoirs = {"en": [], "tr": [], "code": []}
    seen = {key: 0 for key in reservoirs}
    with validation_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            language = row["language"]
            if language not in reservoirs or len(row["text"]) < 80:
                continue
            seen[language] += 1
            item = (row["id"], row["text"])
            if len(reservoirs[language]) < per_language:
                reservoirs[language].append(item)
            else:
                position = rng.randrange(seen[language])
                if position < per_language:
                    reservoirs[language][position] = item
    probes = []
    for language, values in reservoirs.items():
        for index, (document_id, text) in enumerate(values):
            split = max(1, min(len(text) - 1, len(text) // 2))
            probes.append(
                {
                    "id": f"{language}-lm-{index:02d}",
                    "language": language,
                    "category": "heldout_continuation",
                    "document_id": document_id,
                    "prefix": text[:split],
                    "continuation": text[split : split + 160],
                }
            )
    return probes


def build(validation_path: Path, output: Path, seed: int = 20260822) -> dict:
    benchmark = {
        "format_version": 2,
        "benchmark_id": "meaningful-scale-v2",
        "seed": seed,
        "scope": "frozen deterministic development suite; keep a separately held secret suite for final claims",
        "validation_source": {
            "path": str(validation_path),
            "sha256": _sha256(validation_path),
        },
        "multiple_choice": _mc_tasks(seed),
        "generative": _generative_tasks(seed),
        "language_model": _lm_probes(validation_path, seed),
        "permutations": [0, 1, 2, 3],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return benchmark


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/processed/proxy/v2-clean/validation.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/meaningful_scale_v2.json")
    )
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    result = build(args.validation, args.output, args.seed)
    print(
        json.dumps(
            {
                key: len(result[key])
                for key in ("multiple_choice", "generative", "language_model")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
