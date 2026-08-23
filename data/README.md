# Data

Data is organized by lifecycle first, then family and version.

```text
data/
  raw/
    translation/opus100/                 Downloaded upstream corpus
  processed/
    proxy/{v1,v2-clean}/                  Historical flat-text proxy datasets
    distillation/grounded-pilot/v1/       Grounded assistant pilot
    synthetic/pretraining/                Current sharded chat/translation corpus
      shards/
      exports/messages.jsonl
  evaluation/
    tokenizer/tr_morphology.json          Tokenizer-only diagnostic data
```

Files under `raw/` are upstream inputs. Files under `processed/` are generated
training datasets with manifests and split metadata. Files under `evaluation/`
must never be included in tokenizer or model training.

Historical manifests and JSONL `locator` fields intentionally retain their
original source paths and hashes. Those strings are provenance records, not
current filesystem pointers.

## Build proxy v1

```bash
python -m src.data.proxy \
  --config configs/data-generation/proxy/v1.json \
  --output-dir data/processed/proxy/v1
```

## Generate synthetic pretraining data

Local generation:

```bash
python -m src.distillation.synthetic_pretraining \
  --config configs/data-generation/synthetic/pretraining/local-1m.json
```

A100 generation:

```bash
python -m src.distillation.synthetic_pretraining_vllm \
  --config configs/data-generation/synthetic/pretraining/a100-1m.json
```

The currently selected corpus is
`data/processed/synthetic/pretraining/shards/`. It contains both `shard-*.jsonl`
chat data and `translations-*.jsonl` translation data; each record carries its
own train/validation split.
