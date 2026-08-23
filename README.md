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

The active test suite covers only models and tokenization.
