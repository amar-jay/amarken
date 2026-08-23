# Experiments

`experiments/` contains final, machine-readable reports. It mirrors the workflow
layout in `configs/`; mutable checkpoints and step metrics belong in `runs/`.

```text
experiments/
  validation/correctness/
  tokenization/synthetic-pretraining/
  training/
    learning-curves/
    proxy/
    synthetic/<run>/{training,capability}.json
  evaluation/meaningful-scale/
```

Current synthetic evidence:

| Run | Training report | Capability report | Status |
|---|---|---|---|
| CPU smoke | `training/synthetic/cpu-smoke/training.json` | — | completed smoke |
| GPU batch 1, 1M | `training/synthetic/gpu-batch1-1m/training.json` | `training/synthetic/gpu-batch1-1m/capability.json` | completed |
| GPU batch 8, 1M | `training/synthetic/gpu-batch8-1m/training.json` | `training/synthetic/gpu-batch8-1m/capability.json` | completed |

The historical proxy experiments remain under `training/proxy/`. They use
different data/tokenizer contracts and must not be compared by raw loss alone
with the synthetic runs.
