# Data

## Tokenizer reference corpus

The tokenizer sweep uses OPUS-100 v1.0 supervised English–Turkish: 1,000,000
aligned training pairs, 2,000 development pairs, and 2,000 test pairs. Download:

```bash
mkdir -p data/raw/opus100
curl -L \
  https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-en-tr-v1.0.tar.gz \
  -o data/raw/opus100/en-tr-v1.0.tar.gz
tar -xzf data/raw/opus100/en-tr-v1.0.tar.gz -C data/raw/opus100
```

Expected archive SHA-256:
`0d4a941721721c94e99013052491894062d7782efb94a6593a91fd1f0f06cba0`.

OPUS-100 is a sampled mixture of OPUS domains and is suitable for a controlled
tokenizer baseline, not the final pretraining corpus. Training data is used only
for vocabulary learning; untouched dev+test files supply tokenizer metrics.

`tokenizer_eval/tr_morphology.json` is a small hand-curated diagnostic suite,
not training data and not a linguistic gold-standard morphological analyzer.

Run the matched sweep with:

```bash
python -m src.tokenization.sweep \
  --output-dir artifacts/tokenizers \
  --report experiments/tokenizer_sweep.json
```

The three candidates differ only in vocabulary size. They use BPE plus all 256
byte pieces, identity normalization for lossless text preservation, full Unicode
character coverage, fixed special-token IDs, original corpus order, one trainer
thread, and an exact vocabulary-size limit. Fertility is pieces per
whitespace-delimited surface word. Byte fallback rate is emitted `<0xXX>` pieces
divided by all emitted pieces. Turkish morphology fragmentation is measured on
the separate probe set both as pieces per word and as each inflected form's
piece count relative to its lemma. Artifact size includes `.model` and `.vocab`.
Round-trip correctness requires exact string equality after decode(encode(text)).

## Proxy pretraining corpus

Build the non-distilled proxy corpus with:

```bash
python -m src.data.proxy \
  --config configs/proxy_dataset.json \
  --output-dir data/processed/proxy-v1
```

`proxy-v1` starts from deterministic 100k-record samples of each OPUS-100 EN/TR
training side, the project-owned Python modules, and the Python 3.11.15 standard
library under the PSF license. CPython tests, `site-packages`, and `ensurepip` are
excluded: tests resemble evaluation data, while the latter directories introduce
bundled third-party code and licenses. This is a proxy for pipeline/model
validation, not the eventual pretraining mixture; OPUS component licenses must
be resolved individually before redistribution or production training.

Each JSONL record carries stable ID, content hash, source ID, source locator,
language, domain, and split group. Aligned EN/TR rows share a group so translations
cannot cross the train/validation boundary. Selection takes the lowest seeded
hashes instead of the first records. Splitting hashes groups into 100 fixed basis
points (1%) for validation, so results do not depend on traversal order.

Cleaning performs language-local exact deduplication followed by trigram SimHash
near-deduplication at Hamming distance three. Contamination rejects an exact
document or any exact 13-token window shared with configured reference files.
The initial registry covers repository correctness tests and tokenizer probes;
future hidden/public evaluation datasets must be added to
`contamination_references` before any benchmark claim. `manifest.json` records
all source, license, configuration, and output hashes. `contamination.jsonl`
quarantines rejected documents for audit rather than silently deleting them.

The builder accepts only `lines`, `glob`, and `python_stdlib` source kinds. There
is deliberately no synthetic or distillation adapter in proxy-v1.
