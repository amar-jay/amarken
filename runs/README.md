# Runs

`runs/` contains mutable execution output: checkpoints, optimizer state, and
step metrics. It mirrors the workflow hierarchy used by `configs/` and
`experiments/`.

```text
runs/training/
  learning-curves/   Long single-model trajectories
  proxy/             Historical LR screens, scaling runs, and tournaments
  synthetic/         Current synthetic-data training runs
  tokenizer-probes/  Model probes used during tokenizer selection
```

Names ending in `-incomplete` are retained partial runs. They are not completed
evidence and have no matching final experiment report.
