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
