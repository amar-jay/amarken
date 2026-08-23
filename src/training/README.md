# Training

## Correctness gate

Run this before changing model or trainer behavior:

```bash
python -m src.training.correctness \
  --device cpu \
  --output experiments/validation/correctness/gates.json
```

## Synthetic training runs

Original 1M-token baseline (`batch_size=1`, accumulation 4):

```bash
python -m src.training.proxy_experiment \
  --config configs/training/synthetic/gpu-batch1-1m.json \
  --report experiments/training/synthetic/gpu-batch1-1m/training.json
```

Batch-8 throughput comparison (same total exposure, larger update batch):

```bash
python -m src.training.proxy_experiment \
  --config configs/training/synthetic/gpu-batch8-1m.json \
  --report experiments/training/synthetic/gpu-batch8-1m/training.json
```

Long 100-Mi-token run:

```bash
python -m src.training.proxy_experiment \
  --config configs/training/synthetic/gpu-batch1-100mi.json \
  --report experiments/training/synthetic/gpu-batch1-100mi/training.json
```

`proxy_experiment` supports exact per-model resume through `resume_from` in the
configuration. Map a completed arm to its `final.pt` to skip it, and a partial
arm to its latest `step-XXXXXXXX.pt` to continue only the remaining updates.

## Output contract

- `configs/` contains authored inputs.
- `runs/` contains checkpoints and step-level `metrics.jsonl`.
- `experiments/` contains final training and capability reports.

Historical proxy LR screens, scaling runs, and tournaments remain available
under `runs/training/proxy/` and `experiments/training/proxy/`.
