# Configurations

Configurations are grouped by the workflow that consumes them. A filename must
describe the meaningful variant—hardware, physical batch, and token exposure—so
the reader does not need to open several JSON files to distinguish runs.

```text
configs/
  data-generation/
    proxy/v1.json
    synthetic/pretraining/{local-1m,a100-1m,a100-80gb-1m}.json
  tokenization/
    synthetic-pretraining/sweep.json
  training/
    learning-curves/dt-apostrophe/{cpu-smoke,100m}.json
    synthetic/{cpu-smoke,gpu-batch1-1m,gpu-batch8-1m,gpu-batch1-100mi}.json
```

Completed configurations are immutable. A change to data, tokenizer, batch
geometry, schedule, model shape, or exposure gets a new descriptive filename
and `experiment_id`; it does not overwrite the old run's identity.

The current synthetic comparisons are:

| Config | Purpose | Tokens per arm |
|---|---|---:|
| `training/synthetic/cpu-smoke.json` | CPU correctness smoke | 32 |
| `training/synthetic/gpu-batch1-1m.json` | Original GPU baseline | 1,048,576 |
| `training/synthetic/gpu-batch8-1m.json` | Throughput comparison | 1,048,576 |
| `training/synthetic/gpu-batch1-100mi.json` | Long matched run | 104,857,600 |

Reports belong under the matching `experiments/` hierarchy and checkpoints or
metrics under the matching `runs/` hierarchy.
