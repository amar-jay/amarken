# Deterministic proxy evaluation

Run the frozen bilingual suite with:

```bash
python -m src.evaluation.capability \
  --proxy-report experiments/proxy_10m.json \
  --benchmark benchmarks/proxy_capability_v1.json \
  --report experiments/capability_10m.json
```

The benchmark contains three English and three Turkish tasks in each of five
categories. Four fixed choices make accuracy and probability calibration fully
mechanical: each continuation receives teacher-forced log likelihood, the four
scores are normalized, and the highest probability wins. This tests whether a
model prefers the right answer; it does not establish open-ended generation or
chat quality. At 30 tasks, one answer changes aggregate accuracy by 3.33 points,
so this suite is an implementation/progress signal rather than a ranking-grade
benchmark.

Calibration reports multiclass Brier score, correct-choice NLL, and five-bin ECE.
Validation loss covers all packed proxy validation blocks. The contamination
gate scans the exact training JSONL for shared 13-token windows before scoring.
The benchmark path is also registered in future proxy-dataset builds.

Each architecture runs in a fresh one-thread child process. Latency includes
median/p95 batch-one 64-token forwards and median greedy eight-token generation;
the latter uses the common no-cache API and therefore measures current reference
software, not optimized deployment. RAM reports framework-relative resident RSS,
absolute resident RSS, and trainer-checkpoint load peak. Analytical FP16/ternary
deploy bytes, tokenizer bytes, training parameter bytes, and 64-token KV bytes
remain the portable comparison; process RSS includes Python/PyTorch allocators.

## Meaningful-scale v2 development suite

`benchmarks/meaningful_scale_v2.json` is generated deterministically from a
fixed seed and the hashed clean-v2 validation split:

```bash
python -m src.evaluation.build_benchmark_v2
```

It contains 240 EN/TR multiple-choice cases balanced across five categories and
all four original answer positions, 120 EN/TR/code greedy exact-match cases,
and held-out continuation probes for every available language/domain. Every
multiple-choice case is evaluated under four cyclic option permutations.
Reports include Wilson 95% intervals by language/category, all-permutation
accuracy, prediction-content invariance, exact-match accuracy, and continuation
NLL/perplexity.

This is a frozen development suite, not a secret final test. Final capability
claims still require a separately held benchmark whose concrete items and seed
were never available during model or mixture selection.
