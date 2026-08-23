# Data

The active dataset is `processed/synthetic/pretraining/`. Its `shards/`
directory contains synthetic `shard-*.jsonl` records and
`translations-*.jsonl` translation records. `evaluation/tokenizer/` contains
tokenizer-only diagnostics and is not training data.

Proxy data, distillation pilots, and raw upstream translation sources are under
`archive/2026-08-23-pre-reset/data/`.
