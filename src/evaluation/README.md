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
