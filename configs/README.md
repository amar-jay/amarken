## Proxy dataset

`proxy_dataset.json` is the reviewed policy and provenance input to
`python -m src.data.proxy`. Vocabulary/training experiments should identify both
this configuration hash and the generated `manifest.json` hash; the friendly
`proxy-v1` name alone is not sufficient provenance.

The OPUS cap is per language, not combined, to prevent an accidental English
majority. `group_namespace` joins aligned translation rows for leakage-safe
splitting. Adding a source requires an immutable source ID, language/domain,
license assessment, upstream URL, and stable grouping rule. Add benchmark files
to `contamination_references` before running the builder, never afterward.

## Architecture tournament

`lr_screen_10m.json` screens three learning rates independently for named model
variants at context 512. Its report is a required, hashed input to
`proxy_tournament_10m_context512.json`; the tournament refuses to guess a rate
or consume a failed screen. The long configuration uses 512 optimizer updates,
four 512-token micro-batches per update, therefore exactly 1,048,576 training
tokens per variant and seed. `train_token_budget` is larger only to construct a
non-repeating packed pool; it is not the reported consumed-token budget.
