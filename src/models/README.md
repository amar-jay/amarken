# Common model contract

Import experiment-facing APIs from `src.models`. Both current architectures
implement `AmarkenCausalLM` and therefore expose the same:

- `forward(input_ids, attention_mask=None, labels=None) -> CausalLMOutput`;
- `generate(...) -> token_ids` correctness-first autoregressive generation;
- `stats(sequence_length, element_bytes=2) -> ModelStats` parameter, FLOP,
  artifact-byte, training-weight-byte, and KV-cache accounting;
- `save_checkpoint`, `from_checkpoint`, and `restore_training_state` APIs;
- registry-backed `create_config`, `create_model`, `save_config`, `load_config`.

FLOP values are deterministic analytical estimates: learned matrix MACs plus QK
and attention-value MACs. They intentionally exclude normalization, RoPE,
softmax, quantizer bookkeeping, and elementwise operations. Runtime latency and
energy must still be measured on target hardware.

```python
from src.models import create_config, create_model

config = create_config("glimmer", vocab_size=12_000)
model = create_model(config)
print(model.stats(sequence_length=2048))
```
