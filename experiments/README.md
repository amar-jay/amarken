# EN/TR tokenizer sweep

`tokenizer_sweep.json` is the reproducible, machine-readable report for matched
8k, 12k, and 16k SentencePiece BPE tokenizers trained on OPUS-100 EN/TR v1.0.
It records corpus and model hashes, every trainer decision, held-out language
metrics, per-word morphology traces, Unicode stress results, artifact sizes, and
training duration. The report's top-level `passed` gate requires exact requested
vocabulary sizes, no unknown tokens, and no round-trip failures.

The JSON is the source of truth; this file intentionally avoids copying result
values that could become stale after a rerun.

`proxy_10m.json` is the machine-readable DT/Glimmer/Bit smoke-scale tournament.
It contains exact parameter/FLOP accounting, initial/final validation loss,
training loss, throughput, gradient health, Bit ternary statistics and final
checkpoint hashes. Its `passed` field means matching token/update counts only;
it deliberately does not declare a quality winner at this token scale.

`capability_10m.json` evaluates those exact checkpoint hashes on the frozen
`proxy-capability-v1` benchmark and the complete proxy validation split. It
records per-task choice probabilities, category/language accuracy, calibration,
contamination results, isolated-process latency/RAM, KV memory and deployable
artifact accounting. Its pass flag verifies evaluation integrity, not quality.
