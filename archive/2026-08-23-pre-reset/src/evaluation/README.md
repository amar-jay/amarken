# Evaluation

Evaluate the completed synthetic batch-1 baseline with the frozen 30-item EN/TR
suite:

```bash
python -m src.evaluation.capability \
  --proxy-report experiments/training/synthetic/gpu-batch1-1m/training.json \
  --benchmark benchmarks/proxy_capability_v1.json \
  --report experiments/training/synthetic/gpu-batch1-1m/capability.json \
  --device cuda
```

For batch 8, use the corresponding `gpu-batch8-1m` training and capability
paths. The evaluator scans the selected training split for contamination,
scores all 30 multiple-choice tasks, evaluates the validation split, and records
systems measurements. With only 30 four-choice tasks, accuracy is a progress
signal rather than ranking-grade evidence.

The larger development suite is `benchmarks/meaningful_scale_v2.json`. It is a
frozen development benchmark, not a secret final holdout.
