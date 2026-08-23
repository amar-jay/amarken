![amarken](assets/banner.png)

# Amarken

Amarken is currently a deliberately small research workspace containing only:

- decoder model implementations in `src/models/`;
- the EN/TR tokenizer in `src/tokenization/`;
- the synthetic pretraining dataset in `data/processed/synthetic/pretraining/`.

The unsuccessful training/evaluation tournament, its generators, configs,
reports, checkpoints, proxy datasets, and supporting tests were moved intact to
`archive/2026-08-23-pre-reset/`. They are historical material, not part of the
active design.

## Tokenizer

The retained tokenizer is `tiktoken-tr-bpe-12k`:

```bash
python -m src.tokenization.sweep \
  --config configs/tokenization/synthetic-pretraining/sweep.json \
  --evaluate-only
```

Visualize random samples drawn across both `shard-*.jsonl` and
`translations-*.jsonl`:

```bash
python -m src.tokenization.visualize \
  --dataset data/processed/synthetic/pretraining/shards \
  --tokenizer apostrophe=artifacts/tokenizers/v3/tiktoken-tr-bpe-12k.json \
  --samples 5 --legend
```

## Tests

```bash
python -m pytest
```

## Training

Training streams the chat shards, supervises assistant responses only, and packs
conversations with isolated attention segments. PyTorch Lightning provides mixed
precision, distributed execution, gradient accumulation/clipping, validation,
CSV metrics, and resumable checkpoints:

```bash
python -m src.training.trainer \
  --config configs/synthetic_training_cpu_smoke.json \
  --model dt
```

Resume with `--resume runs/synthetic-proxy-cpu-smoke/dt/checkpoints/last.ckpt`.

### Experiment tracking

Every optimizer update writes a structured record to
`runs/.../step_audit.jsonl`. It includes the weighted loss, learning rates,
token/segment counts, parameter health, and gradient norms before and after
clipping. Lightning logs the compact metrics to CSV and W&B.

The smoke configuration uses W&B offline mode, so it needs no credentials. To
inspect or upload that run later:

```bash
wandb sync runs/synthetic-proxy-cpu-smoke/dt/wandb/offline-run-*
```

For an online run, authenticate once with `wandb login`, then set
`wandb_mode` to `"online"` in the training config. Set `wandb_log_model` to
`true` only in online mode to upload Lightning checkpoints as W&B artifacts.

---

Surprising result: tr-biased is not beating vanilla; it's $~2.1%$ worse in **v3** tokenizer.
