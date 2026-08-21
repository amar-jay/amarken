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
