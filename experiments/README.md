# EN/TR tokenizer sweep

`tokenizer_sweep.json` is the reproducible, machine-readable report for matched
8k, 12k, and 16k SentencePiece BPE tokenizers trained on OPUS-100 EN/TR v1.0.
It records corpus and model hashes, every trainer decision, held-out language
metrics, per-word morphology traces, Unicode stress results, artifact sizes, and
training duration. The report's top-level `passed` gate requires exact requested
vocabulary sizes, no unknown tokens, and no round-trip failures.

The JSON is the source of truth; this file intentionally avoids copying result
values that could become stale after a rerun.
